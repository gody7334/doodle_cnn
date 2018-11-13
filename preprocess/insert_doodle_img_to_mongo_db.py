import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utils.common import *
from utils.mongodb_conn import *
from utils.mongodb_conn import _connect_mongo
from preprocess.dataloader import drawing_to_image
from keras.preprocessing.image import array_to_img
from imageio import imread
from io import BytesIO
import io
import base64
from joblib import Parallel, delayed



def convert_to_base64(np_img):
    if len(np_img.shape) == 2:
        img = array_to_img(np.stack((np_img,)*3, -1),data_format='channels_last',scale=True)
    elif len(np_img.shape) == 3:
        img = array_to_img(np_img, data_format='channels_last',scale=True)
    else:
        raise 'wrong img format'

    img = img.convert('LA')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def check_img_encoding():
    df = read_mongo(
        db='dataset',
        collection='doodle_quickdraw',
        query={'key_id': 6100521775529984},
        project={'key_id':1, 'drawing':1, 'word':1, 'img':1},
        host='localhost',
        port=27017,
        username=None,
        password=None,
        no_id=True,
    )

    img = np.array(imread(io.BytesIO(base64.b64decode(df['img'].values[0])),as_gray=True))
    overlay=255-img
    image_show('overlay',overlay, resize=2)
    cv2.waitKey(0)


def insert_img_encoding_to_db():
    ids = list(pd.read_csv(f'../../data/train_key_id_CV/all/train_df_0.csv').key_id.values)
    ids = ids + list(pd.read_csv(f'../../data/train_key_id_CV/all/val_df_0.csv').key_id.values)

    def insert_img_encoding(key_id):
        df = read_mongo(
            db='dataset',
            collection='doodle_quickdraw',
            query={'key_id': key_id},
            project={'key_id':1, 'drawing':1, 'word':1},
            host='localhost',
            port=27017,
            username=None,
            password=None,
            no_id=True,
        )
        drawing = df['drawing'].values[0]
        drawing = eval(drawing)
        image = drawing_to_image(drawing, 128, 128)
        set_data('dataset', 'doodle_quickdraw', {'key_id': key_id}, {'img':  convert_to_base64(image)})
        print(key_id,flush=True)

    Parallel(n_jobs=4)(delayed(insert_img_encoding)(id.item()) for id in ids)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    insert_img_encoding_to_db()
    # check_img_encoding()
