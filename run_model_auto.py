import subprocess
import sys
import yaml
import requests
import argparse


# def reminder(error_msg):
#     notice_url = "https://www.feishu.cn/flow/api/trigger-webhook/bf319c6b87ffeedec308017b424257d7"
#     payload = {
#         "msg": error_msg,
#     }
#     requests.post(url=notice_url, data=payload)

def change_yaml_value(file_path: str, value: int, new_file_path: str = None):
    with open(file_path, 'r', encoding="UTF-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config["testing"]["ckpt_epoch"] = int(value)
    if new_file_path:
        file_path = new_file_path
    with open(file_path, 'w', encoding="UTF-8") as file:
        yaml.dump(config, file)

def run_command(yaml_value, new_file_path):
    # 定义命令行参数
    cuda_env = "CUDA_VISIBLE_DEVICES=1"
    script_name = "srun_gpt_all.sh"
    config_file = "configs/cc_motion_gpt.yaml"
    mode = "eval"
    value = "1"

    # 更新YAML文件
    if new_file_path != config_file:
        change_yaml_value(config_file, yaml_value, new_file_path)
    else:
        change_yaml_value(config_file, yaml_value)

    # 拼接命令
    command = f"{cuda_env} sh {script_name} {config_file} {mode} {value}"
    # command = f"python test.py"

    try:
        # 执行命令
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("命令执行成功！")
        print("输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("程序异常退出，报错信息如下：")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update YAML value and execute CUDA")
    parser.add_argument("--ckpt_epoch", type=int, help="Update ckpt_epoch value in the YAML file.")
    parser.add_argument("--new_file_path", type=str, help="The path to the new YAML file.", default="configs/cc_motion_gpt.yaml")
    args = parser.parse_args()

    run_command(args.ckpt_epoch, args.new_file_path)
