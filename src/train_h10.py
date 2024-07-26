#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>

import argparse
import json
import os, sys
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import time
import tqdm
import sys

if tf.VERSION >= '2':
    tf.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False,
                                                  'constant_folding': False,
                                                  'shape_optimization': False,
                                                  'remapping': False,
                                                  'arithmetic_optimization': False,
                                                  'dependency_optimization': False,
                                                  'loop_optimization': False,
                                                  'disable_meta_optimizer': True
                                                  })


import model, sample, encoder
from load_dataset import load_dataset, Sampler
import os
import glob
CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'


parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='124M', help='Pretrained model name')
parser.add_argument('--models_dir', metavar='PATH', type=str, default='models', help='Path to models directory')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--encoding', type=str, default='utf-8', help='Set the encoding for reading and writing files.')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--twremat', default=False, action='store_true', help='Use tensor rematerialization (better than memory_saving_gradients and works with tensorflow 2.0).')
parser.add_argument('--twremat_memlimit', type=str, default='12G', help='Memory usage limit/target for twremat. Can be an integer, or an integer suffixed with K/M/G for kilo/mega/giga-bytes.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd>.')
parser.add_argument('--noise', type=float, default=0.0, help='Add noise to input training data to regularize against typos.')

parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
#parser.add_argument('--run_name2', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--run_name2', type=str, default='run_part1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=1000, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=200, help='Write a checkpoint every N steps')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=2, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=40, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=10, help='Calculate validation loss every STEPS steps.')

parser.add_argument('--part', metavar='Part', type=str, default=all, help='whether load all-weight')
parser.add_argument('--detel', metavar='DEL', type=int, default=60, help='detel checkpoint ')
def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context


def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name, models_dir=args.models_dir)
    hparams = model.default_hparams()
    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)
    total_run_time = 0
    max_run_time = 3 * 3600 
    max_checkpoints_to_keep = 3
   


    
    with tf.Session() as sess:
        # Fully static shape required to make memory accounting in
        # twremat accurate.
        train_context = tf.placeholder(tf.int32, [args.batch_size, 1024])
        train_context_in = randomize(train_context, hparams, args.noise)
        train_output = model.model(hparams=hparams, X=train_context_in)
        train_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=train_context[:, 1:], logits=train_output['logits'][:, :-1]))

        if args.val_every > 0:
            val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
            val_output = model.model(hparams=hparams, X=val_context)
            val_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
            val_loss_summary = tf.summary.scalar('val_loss', val_loss)

        sample_context = tf.placeholder(tf.int32, [args.batch_size, None])
        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.sample_length,
            context=sample_context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=args.top_k,
            top_p=args.top_p)
  #      all_s = tf.trainable_variables()
        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars
####

        #variable_names = []
        #with open('all-layer.txt', 'r') as fp:
       #     variable_names = fp.read().splitlines()
        var_list1 = {}
  #      key =['attn/c_attn/w','ln_1/g','attn/c_proj/w','mlp/c_fc/w']
     #   key =['wpe','wte','attn','ln_1','ln_2','mlp','ln_f']
        key =['wpe','attn','mlp']
        var_list1 =  [v for v in all_vars if any(keyword in v.name for keyword in key)]
      #  var_list1 =[v for v in all_vars if 'model' in v.name] 
      #  print(var_list1)
        part_var1= [v for v in var_list1 if 'h10' not in v.name]
        part_var2= [v for v in part_var1 if 'h11' not in v.name]
        part_vars= [v for v in part_var2 if 'h1' not in v.name]
        with open('l-layer.txt', 'w') as fp:
            for var in part_vars:
                fp.write(var.name + '\n')
      #  var_list1 = [v for v in all_vars if any(name in v.name for name in variable_names)] 
    ####
        if args.optimizer == 'adam':
            print('Using Adam optimizer', file=sys.stderr)
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'sgd':
            print('Using SGD optimizer', file=sys.stderr)
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        else:
            exit('Bad optimizer:', args.optimizer)

        if args.memory_saving_gradients:
            if tf.VERSION >= '2':
                exit('Memory saving gradients are not supported in tensorflow 2.x')
            import memory_saving_gradients
            opt_grads = memory_saving_gradients.gradients(train_loss, train_vars)
        elif args.twremat:
            import tfremat
            opt_grads = tf.gradients(train_loss, train_vars)
            (train_loss, opt_grads) = tfremat.tf_remat((train_loss, opt_grads), memlimit=args.twremat_memlimit)
        else:
            opt_grads = tf.gradients(train_loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', train_loss)

        # if args.twremat:
        #     import tfremat
        #     # Applying tfremat to opt_apply has more accurate
        #     # accounting but is a bit iffier since side effecting ops
        #     # have more restrictions for correctness. If in doubt
        #     # revert back to version using opt_grads above.
        #     (opt_apply, train_loss, summary_loss) = (
        #         tfremat.tf_remat((opt_apply, train_loss, summary_loss), memlimit=args.twremat_memlimit))


        summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))


        sess.run(tf.global_variables_initializer())
        ckp = tf.train.latest_checkpoint(os.path.join('models', args.model_name))
    #    saver.restore(sess, ckp)  
        if args.part=="all":
            ckpt = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, args.run_name))
        else:
            ckpt = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, args.run_name2))

        if ckpt is None:
            
        
            print("正常初始化运行")
        else:

            if args.part=="all":
                checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.run_name)
           # ckpt_file = os.path.join(CHECKPOINT_DIR, args.run_name,'model-1000')
                with open(os.path.join(checkpoint_dir, 'counter'), 'r') as counter_file:
                    counter = int(counter_file.read())
                checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
                checkpoint_file = checkpoint_prefix + '-' + str(counter)

                saver = tf.train.Saver(
                var_list=all_vars,
                max_to_keep=1,
                keep_checkpoint_every_n_hours=2)
                saver.restore(sess,checkpoint_file) 
                print("load all weight",checkpoint_file)
            elif args.part=="part":
                checkpoint_dir2 = os.path.join(CHECKPOINT_DIR, args.run_name2)
                with open(os.path.join(checkpoint_dir2, 'counter'), 'r') as counter_file:
                    counter2 = int(counter_file.read())
                checkpoint_prefix2 = os.path.join(checkpoint_dir2, 'model')
                checkpoint_file = checkpoint_prefix2 + '-' + str(counter2)
               # ckpt_file2= os.path.join(CHECKPOINT_DIR, args.run_name2,'model-3000')
               
                saver = tf.train.Saver(
                var_list=var_list1,
                max_to_keep=1,
                keep_checkpoint_every_n_hours=2)
              
                
               # saver.restore(sess,ckpt_file)   
                saver.restore(sess, checkpoint_file)
                print("load part weight",checkpoint_file)

               










        print('Loading dataset...')
        chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
        data_sampler = Sampler(chunks)
        if args.val_every > 0:
            if args.val_dataset:
                val_chunks = load_dataset(enc, args.val_dataset, args.combine, encoding=args.encoding)
            else:
                val_chunks = chunks
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')


     #记录训练的开始时间
        all_start_time = time.time()


        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_chunks, seed=1)
            val_batches = [[val_data_sampler.sample(1024) for _ in range(args.val_batch_size)]
                           for _ in range(args.val_batch_count)]

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name2, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver = tf.train.Saver(
       #         var_list=var_list1,
                var_list=all_vars,
                max_to_keep=1,
                keep_checkpoint_every_n_hours=2)

            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def save2():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name2))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name2,
                             'model-{}').format(counter))
            saver = tf.train.Saver(
                var_list=var_list1,
             #   var_list=all_vars,
                max_to_keep=1,
                keep_checkpoint_every_n_hours=2)

            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name2, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')        

#只保存最大数量的checkpoint
     #   def delete_old_checkpoints():
     #       if args.part=="all":
     #           checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.run_name)
     #       elif args.part=="part":
     #           checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.run_name2)
      #      checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'model-*'))
    #        checkpoint_file = [file for file in checkpoint_files if not file.endswith(('.index', '.meta'))]
       #     checkpoint_files.sort(key=os.path.getctime)
       #     files_to_delete = checkpoint_files[:-max_checkpoints_to_keep]
       #     for file_to_delete in files_to_delete:
       #         os.remove(file_to_delete)






        def generate_samples():
            print('Generating samples...')
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={sample_context: args.batch_size * [context_tokens]})
                for i in range(min(args.sample_num - index, args.batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            print(text)
            maketree(os.path.join(SAMPLE_DIR, args.run_name))
            with open(
                    os.path.join(SAMPLE_DIR, args.run_name,
                                 'samples-{}').format(counter), 'w', encoding=args.encoding) as fp:
                fp.write('\n'.join(all_text))

        def validation():
            print('Calculating validation loss...')
            losses = []
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss, feed_dict={val_context: batch}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            print(
                '[{counter} | {time:2.2f}] validation loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))

        def sample_batch():
            return [data_sampler.sample(1024) for _ in range(args.batch_size)]


        avg_loss = (0.0, 0.0)
        start_time = time.time()

        # print('Evaluating grads..')
        # tf2.profiler.experimental.start('logdir')
        # sess.run((opt_apply, train_loss, summaries), feed_dict={train_context: sample_batch()})
        # tf2.profiler.experimental.stop()
        # print('Succeeded')
        # exit()

        try:



            while True:
               # physical_devices = tf2.config.experimental.list_physical_devices('GPU')

                    # 打印 GPU 设备数量
               # print("Number of GPUs available: ", len(physical_devices))
                
                if counter % args.save_every == 0:
                    df=time.time() - start_time
                    save2()
                    dt=time.time() - start_time
                    dv=dt-df
                #    save2()
                #    fh=time.time()-start_time
                 #   fht=fh-dt
                    print(f"time of saving checkpoint:{dv:.2f}")
                  #  print(f"time of saving checkpoint_part:{fht:.2f}")

                if counter % args.sample_every == 0:
                    generate_samples()
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    validation()
              #  if counter % args.detel == 0:
                #    delete_old_checkpoints()
            
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, train_loss, summaries),
                    feed_dict={train_context: sample_batch()})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)
            #    dk=time.time()-start_time
            #    print(f"dk:{dk:.2f}")
                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time ,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))
                
                
                times=time.time() - start_time
                counter += 1
            #    if counter==5784:
                if counter==2100:

                    sys.exit()



           
        except KeyboardInterrupt:
            print('interrupted')


           # save()
           # save2()


if __name__ == '__main__':
    main()

