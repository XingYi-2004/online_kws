from omegaconf import OmegaConf
from aligo import Aligo
from pathlib import Path

def init(conf):
    ali = Aligo()  # 第一次使用，会弹出二维码，供扫描登录
    
    user = ali.get_user()  # 获取用户信息
    print(user.user_name, user.nick_name, user.phone)  # 打印用户信息
    
    ll = ali.get_file_list()  # 获取网盘根目录文件列表
    for file in ll:  # 遍历文件列表
        print(file.file_id, file.name, file.type)  # 打印文件信息
       
def upload(conf):
    ali = Aligo()
    remote_folder = ali.get_folder_by_path(conf.remote)
    local_path = Path(conf.local)
    if local_path.is_dir(): # e.g. data/DNS2020
        ali.upload_folder(conf.local, parent_file_id=remote_folder.file_id)
    elif local_path.is_file(): # e.g. data/DNS2020/train.hdf5
        ali.upload_files([conf.local], parent_file_id=remote_folder.file_id)
    elif local_path.parent.is_dir(): # e.g. data/DNS2020/*.csv
        file_list = [str(item) for item in local_path.parent.glob(local_path.name)]
        ali.upload_files(file_list, parent_file_id=remote_folder.file_id)
    else:
        print("local error:", conf.local)

def download(conf):
    ali = Aligo()
    remote_path = Path(conf.remote)
    if remote_path.name.startswith("*"): # e.g. workspace/data/DNS2020/*.csv
        folder = ali.get_folder_by_path(str(remote_path.parent), create_folder=False)
        file_list = ali.get_file_list(parent_file_id=folder.file_id)
        files = []
        for file in file_list:
            if file.name.endswith(remote_path.suffix): # matched
                files.append(file)
        ali.download_files(files, local_folder=conf.local)
    else:
        file = ali.get_file_by_path(conf.remote)
        if file: # download file, e.g. workspace/data/DNS2020/test.hdf5
            ali.download_file(file=file, local_folder=conf.local)
        else: # download floder, e.g. workspace/data/DNS2020
            folder = ali.get_folder_by_path(conf.remote, create_folder=False)
            ali.download_folder(folder.file_id, local_folder=conf.local)
    
if __name__ == '__main__':
    conf = OmegaConf.create({         
         "cmd": "download",
         "local": "data",
         "remote": "data"})
    conf.merge_with_cli()    
    eval(conf.cmd)(conf)