import os  # 导入os模块


def search_file(start_dir):
    img_list = []
    extend_name = ['.jpg', '.JPEG', '.jpeg']  # image format defined by file endings
    os.chdir(start_dir)  # 改变当前工作目录到指定的路径

    for each_file in os.listdir(os.curdir):
        # listdir()返回指定的文件夹包含的文件或文件夹的名字的列表 curdir表示当前工作目录

        img_prop = os.path.splitext(each_file)
        if img_prop[1] in extend_name:
            img_list.append(os.getcwd() + os.sep + each_file + os.linesep)
            # os.getcwd()获得当前路径 os.sep分隔符 os.linesep换行符

        if os.path.isdir(each_file):  # isdir()判断是否为文件夹
            search_file(each_file)  # 递归搜索子文件夹下的图片
            os.chdir(os.pardir)  # back to last directory

    with open(r'/mcunet/jobs/img_list.txt', 'a') as file_obj:  # output file's location and name
        file_obj.writelines(img_list)  # writelines. write data by line


if __name__ == '__main__':
    start_dir_list = [r'/home/ytwanghaoyu/PycharmProjects/tinyml/mcunet/dataset/imagenet/val',]
    for each_dir in start_dir_list:
        search_file(each_dir)