import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utils.common import *
from utils.mongodb_conn import *
from utils.mongodb_conn import _connect_mongo



CLASS_NAME=\
['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'airplane', 'alarm_clock', 'ambulance', 'angel',
 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn',
 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee',
 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel',
 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup',
 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship',
 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser',
 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses',
 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo',
 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan',
 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger',
 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck',
 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant',
 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop',
 'leaf', 'leg', 'light_bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox',
 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito',
 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean',
 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants',
 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano',
 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond',
 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'river', 'roller_coaster', 'rollerskates', 'sailboat',
 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw',
 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear',
 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle',
 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch',
 'yoga', 'zebra', 'zigzag']


#small dataset for debug
# CLASS_NAME = ['apple','bee', 'cat', 'fish', 'frog', 'leaf']
# DATA_DIR = '../../data/train_key_id_CV/all/'

NUM_CLASS = len(CLASS_NAME)
TRAIN_DF  = []
TEST_DF   = []




def null_augment(drawing,label,index):
    cache = Struct(drawing = drawing.copy(), label = label, index=index)
    image = drawing_to_image(drawing, 128, 128)
    return image, label, cache

# how to stack each data into batch
def null_collate_new(batch):
    cache = batch[0][2]
    input = batch[0][0]
    truth = batch[0][1]
    input = torch.from_numpy(input).float()

    if truth[0] is not None:
        truth = np.array(truth)
        truth = torch.from_numpy(truth).long()

    return input, truth, cache

def null_collate(batch):
    batch_size = len(batch)
    cache = []
    input = []
    truth = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        cache.append(batch[b][2])

    input = np.array(input).transpose(0,3,1,2)
    input = torch.from_numpy(input).float()

    if truth[0] is not None:
        truth = np.array(truth)
        truth = torch.from_numpy(truth).long()

    return input, truth, cache

#----------------------------------------

def drawing_to_image(drawing, H, W):

    point=[]
    time =[]
    for t,(x,y) in enumerate(drawing):
        point.append(np.array((x,y),np.float32).T)
        time.append(np.full(len(x),t))

    point = np.concatenate(point).astype(np.float32)
    time  = np.concatenate(time ).astype(np.int32)

    #--------
    image  = np.full((H,W,3),0,np.uint8)
    x_max = point[:,0].max()
    x_min = point[:,0].min()
    y_max = point[:,1].max()
    y_min = point[:,1].min()
    w = x_max-x_min
    h = y_max-y_min
    #print(w,h)

    s = max(w,h)
    norm_point = (point-[x_min,y_min])/s
    norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)


    #--------
    T = time.max()+1
    for t in range(T):
        p = norm_point[time==t]
        x,y = p.T
        image[y,x]=255
        N = len(p)
        for i in range(N-1):
            x0,y0 = p[i]
            x1,y1 = p[i+1]
            cv2.line(image,(x0,y0),(x1,y1),(255,255,255),1,cv2.LINE_AA)

    return image

class DoodleDataset(Dataset):

    def __init__(self, mode, cv_file='<NIL>', augment = null_augment, complexity = 'simplified', bsize=1):
        super(DoodleDataset, self).__init__()
        assert complexity in ['simplified', 'raw']
        start = timer()

        self.cv_file    = cv_file
        self.augment    = augment
        self.mode       = mode
        self.complexity = complexity
        self.bsize = bsize

        self.df     = []
        self.id     = []

        if mode=='train':
            # countrycode, drawing, key_id, recognized, timestamp, word
            self.id = pd.read_csv(f'{DATA_DIR}{cv_file}.csv').key_id.values
            print('\r\t load split: %24s  %s'%(cv_file,time_to_str((timer() - start),'sec')),end='',flush=True)
            print('')

        if mode=='test':
            global TEST_DF
            # key_id, countrycode, drawing
            TEST_DIR='/home/gody7334/Project/tensorflow/ipython/doodle-quickdraw/data/test'
            if TEST_DF == []:
                TEST_DF = pd.read_csv(TEST_DIR + '/test_%s.csv'%(complexity))
                self.id = np.arange(0,len(TEST_DF))

            self.df = TEST_DF

        print('')

    def __str__(self):
        N = len(self.id)
        string = ''\
        + '\tcv_file        = %s\n'%self.cv_file \
        + '\tmode         = %s\n'%self.mode \
        + '\tcomplexity   = %s\n'%self.complexity \
        + '\tlen(self.id) = %d\n'%N \
        + '\n'
        return string

    def __getitem__new(self, index):
        #TODO read_mongo from a batch of key_id
        if self.mode=='train':
            key_id = self.id[index].item()
            df = sample_mongo(
                db='dataset',
                collection='doodle_quickdraw',
                query={},
                project={},
                host='localhost',
                port=27017,
                username=None,
                password=None,
                no_id=True,
                num_sample=self.bsize
            )
            # img = np.array(imread(io.BytesIO(base64.b64decode(df['img'].values[0])),as_gray=True))
            df['img'] = df['drawing'].apply(lambda x: drawing_to_image(eval(x),128,128))
            df['label'] = df['word'].apply(lambda x: CLASS_NAME.index(x.replace(' ','_')))
            img = np.stack(df['img'].values,axis=0).transpose(0,3,1,2)
            label = np.stack(df['label'].values,axis=0)
            cache = Struct(img=img, label=label)
            return (img, label, cache)

        if self.mode=='test':
            label=None
            drawing = self.df['drawing'][index]
            drawing = eval(drawing)

        return self.augment(drawing, label, index)

    def __getitem__(self, index):

        if self.mode=='train':
            key_id = self.id[index].item()
            df = read_mongo(
                db='dataset',
                collection='doodle_quickdraw',
                query={'key_id': key_id},
                project={},
                host='192.168.0.7',
                # host='localhost',
                port=27017,
                username=None,
                password=None,
                no_id=True,
            )
            # img = np.array(imread(io.BytesIO(base64.b64decode(df['img'].values[0])),as_gray=True))
            drawing = df['drawing'].values[0]
            drawing = eval(drawing)
            label = CLASS_NAME.index(df['word'].values[0].replace(' ','_'))
            # return (img, label, Struct(img=img, label=label, index=index))


        if self.mode=='test':
            label=None
            drawing = self.df['drawing'][index]
            drawing = eval(drawing)

        return self.augment(drawing, label, index)

    def __len__(self):
        return len(self.id)


# check #################################################################
def run_check_train_data():

    dataset = DoodleDataset('train', 'train_df_0')
    print(dataset)

    ipdb.set_trace()
    #--
    num = len(dataset)
    for m in range(num):
        #i = m
        i = np.random.choice(num)
        image, label, cache = dataset[i]

        print('%8d  %8d :  %3d    %s'%(i,cache.index,label,CLASS_NAME[label]))

        overlay=255-image
        image_show('overlay',overlay, resize=2)
        cv2.waitKey(0)

def run_check_test_data():

    dataset = DoodleDataset('test')
    print(dataset)

    #--
    num = len(dataset)
    for m in range(num):
        i = m
        #i = np.random.choice(num)
        image, label, cache = dataset[i]

        print('%8d  %8d : '%(i,cache.index))

        overlay=255-image
        image_show('overlay',overlay, resize=2)
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_data()
    #run_check_test_data()

