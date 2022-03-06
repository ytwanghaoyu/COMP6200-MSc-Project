import os  # 导入os模块


def search_file(start_dir):
    txt_list = []
    extend_name = ['.txt', '.TXT']  # image format defined by file endings
    os.chdir(start_dir)  # 改变当前工作目录到指定的路径

    for each_file in os.listdir(os.curdir):
        # listdir()返回指定的文件夹包含的文件或文件夹的名字的列表 curdir表示当前工作目录

        txt_prop = os.path.splitext(each_file)
        if txt_prop[1] in extend_name:
            txt_list.append(os.getcwd() + os.sep + each_file)
            # os.getcwd()获得当前路径 os.sep分隔符 os.linesep换行符

    return txt_list

def convert(txt_list):
    print(txt_list)
    for i in range(len(txt_list)):
        # open file
        f = open(str(txt_list[i]), "r", encoding='utf8')
        with open(str(txt_list[i][:-4])+'1'+'.txt', 'a') as f2:
            # bian li wenjian,bing xie ru xin wen jian
            for line in f:
                line = line.strip('\n')
                f2.write(line + " ")

            # close file
            f2.close()
            f.close()



if __name__ == '__main__':
    start_dir_list = r'/home/ytwanghaoyu/Downloads'
    txt_list=search_file(start_dir_list)
    convert(txt_list)