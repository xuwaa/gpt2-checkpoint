import os
import time
import shutil
import multiprocessing
import subprocess
# 监控的目录和目标目录
source_dir = "/dev/shm/a/checkpoint/medium"  # 要监控的目录
target_dir = "/home/users/u0001456/pytorch/project/gpt-2/checkpoint/medium2"  # 目标目录
#source_dir="/home/users/u0001456/pytorch/project/gpt-2/checkpoint/run1"
# 函数用于监控目录并定期移动文件
def monitor_and_move_files(source_dir, target_dir, interval):
    while True:
        try:
            # 列出目录下的所有文件
            
            if os.path.exists(source_dir):
                files = os.listdir(source_dir)
                counter_path = os.path.join(source_dir, 'counter')
                with open(counter_path, 'r') as fp:
                    checkpoint_value = int(fp.read().strip())
            keep_files =[f for f in files if f.startswith('model-') ]
            for file in files:
                if file  in keep_files:
                    source_file_path = os.path.join(source_dir, file)
                    target_file_path = os.path.join(target_dir, file)
                    shutil.move(source_file_path, target_file_path)

        #    for file in files:
        #        file_path = os.path.join(source_dir, file)
                # 在这里检查文件大小或其他条件，然后决定是否要移动文件
                # 你可以使用os.path.getsize()来获取文件大小
        #        if os.path.exists(file_path) and not file.endswith(".tempstate"):
                # 假设你要移动文件，你可以使用shutil.move()来执行移动操作
                   # shutil.move(file_path, os.path.join(target_dir, file))
         #           shutil.remove(os.path.join(target_dir, file))
        except Exception as e:
            # 处理任何异常，例如文件被占用等
            print(f"Error: {e}")

        time.sleep(interval)

# 函数用于生成文件
def generate_files():
    command = "python src/train1.py --dataset data2 --model_name 345M --models_dir models --part all"
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    # 创建并启动监控进程
    interval = 80  # 每隔10秒监控一次
    monitor_process = multiprocessing.Process(target=monitor_and_move_files, args=(source_dir, target_dir, interval))
    monitor_process.start()


    try:
        # 同时启动生成文件的功能
        generate_files()

        # 主程序可以在这里添加其他逻辑
    except KeyboardInterrupt:
        # 当主程序结束时，通过捕捉 KeyboardInterrupt 异常来停止监控进程
        stop_event.set()
        monitor_process.join()  # 等待监控进程结束

    # 主程序执行完成后，监控进程也会停止
