import numpy as np
import glob
from keras.preprocessing.image import load_img, img_to_array
import re
import os


def load_images_from_label_folder(path, img_width, img_height, train_test_ratio=(9,1)):
    """
    指定されたディレクトリの中の画像をすべて探して，ディレクトリ名に基づいてタグ付けする
    :param path: 画像が入ったディレクトリ
    :param img_width: 画像の幅
    :param img_height: 画像の高さ
    :param train_test_ratio: ？
    :return:
    """

    # ----- ラベルを作成する -----
    # 画像が入ったディレクトリを指定する．
    # unix:     "./images/*"
    # windows:  ".\\images\\*"
    input_img_path = os.path.join(path, "*")

    # フォルダ内のファイル一覧を取得
    data_list = glob.glob(input_img_path)
    print('data_list', data_list)

    paths_and_labels = []

    # ファイルを作成して開きながら
    with open("who_is_who.txt", "w") as f:

        # リストをインデックス付きで回す
        for (i, data_folder_name) in enumerate(data_list):

            paths_and_labels.append([data_folder_name, i])

            # 正規表現で欲しいパスのパターンを定義
            pattern = r".*/(.*)"

            # 定義したパータンに一致するものをファイルに書く
            match_ob = re.finditer(pattern, data_folder_name)
            directory_name = ""
            if match_ob:
                for a in match_ob:
                    directory_name += a.groups()[0]
            f.write(directory_name + "," + str(i) + "\n")

    # ----- 訓令用データである画像を取得する -----
    all_train_img_data = []

    # すべての画像のパスをリストに入れる
    for path_and_label in paths_and_labels:
        path, label = path_and_label
        image_list = glob.glob(path + '/*')

        print(path)

        for img_name in image_list:
            all_train_img_data.append((img_name, label))

    all_train_img_data = np.random.permutation(all_train_img_data)

    train_x_list = []
    train_y_list = []

    print(all_train_img_data)

    for (img_path, label) in all_train_img_data:

        print(img_path)

        img = load_img(img_path, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        train_x_list.append(img_array)
        train_y_list.append(label)

    threshold = (train_test_ratio[0]*len(train_x_list))//(train_test_ratio[0]+train_test_ratio[1])
    test_x = np.array(train_x_list[threshold:])
    test_y = np.array(train_y_list[threshold:])
    train_x_list = np.array(train_x_list[:threshold])
    train_y_list = np.array(train_y_list[:threshold])

    return (train_x_list, train_y_list), (test_x, test_y)


if __name__ == '__main__':

    (train_x, train_y), (_, _) = load_images_from_label_folder('./images', 128, 128)
    print('trainx.shape:', train_x.shape)
