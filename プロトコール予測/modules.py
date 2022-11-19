#日本語が混じっている単語のリスト作成
import re
from transformers import BertJapaneseTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import fasttext 
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import models, losses, evaluation, SentenceTransformer
import mojimoji
from sklearn.utils import class_weight

import lightgbm as lgb

#onehotをつなげる
def one_hot(df, col):
    categories = df[col].unique()
    print(categories)
    df[col] = pd.Categorical(df[col], categories=categories)
    df_c = pd.get_dummies(df[col])
    df = pd.concat([df, df_c], axis=1)
    del df[col]

    return df

#TextAugumententiaon

#dfとラベルを入れると目的が表示される。同じラベルごとのdfが2つ出力される
def ShowList(df, label):
    df = df[df['label']==label].reset_index(drop=True)
    print(len(df))
    for i in range(len(df)):
        print(i+1)
        print (df.loc[i]['purpose'])
        print("=")
        #df2 = df.copy()
    
    return df

#目的を書き換えたリストを2つとdfを2つ入力で合わせたdfが出力される。
#年齢は半分はクラスごとの平均の正規分布からランダムに変更
def makeAug(df, List, YC=True):
    df2 = df.copy()
    #df.loc[:,'purpose']= List1
    #df2.loc[:,'purpose']= List2
    num=int(df2['year'].mean())
    generator = np.random.default_rng()
    rnd = generator.normal(loc=num, scale=10, size=len(df))
    if YC==True:
        df2.loc[:,'year']=rnd
        

    df = pd.concat([df, df2], ignore_index=True)
    df['year'] = df['year'].round().astype('int')
    df.loc[:,'purpose']= List
    
    return df


#わかちがき        
def wakachi(Str):
    import MeCab
    stop_words=  [',','｡','.','右','左','*','(',')','委任',':','。','、',',','.','+']
    tagger = MeCab.Tagger(r"-d /var/lib/mecab/dic/ipadic-utf8/ -u dic/MANBYO_201907_Dic-utf8.dic -Owakati")              
    result = tagger.parse(Str).split()
    words = []
    stop_words = stop_words
    for word in result:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        if word in stop_words: # ストップワードに含まれるものは除外
            continue
        if len(word) < 1: #  1文字、0文字（空文字）は除外
            continue
        words.append(word)
    words = ' '.join(words)
    return words

#名詞のみ
def meishi(text):
    import MeCab
    stop_words= [',','｡','.','右','左','両側','*','(',')','委任',':','。','、',',','.','+','疑い','･']
    mecab = MeCab.Tagger(r"-d /var/lib/mecab/dic/ipadic-utf8/ -u dic/MANBYO_201907_Dic-utf8.dic")
    result = mecab.parse(text)
    lines = result.split('\n')
    nounAndVerb = []#「名詞」と「動詞」を格納するリスト
    for line in lines:
        feature = line.split('\t')
        if len(feature) == 2: #'EOS'と''を省く
            info = feature[1].split(',')
            hinshi = info[0]
            if hinshi in ('名詞', '動詞'):
                nounAndVerb.append(feature[0])
    words = []
    for word in nounAndVerb:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        if word in stop_words: # ストップワードに含まれるものは除外
            continue
        if len(word) < 1:#  1文字、0文字（空文字）は除外
            continue
        words.append(word)

    #words = ' '.join(words)
    return(words)

# fastTextでベクトル化する
# モデルを入れて、インスタンス化しVectrizerで単語リストを入れるとベクトルが平均されて出力される
class FastText_Vectrizer:
    def __init__(self, model):
        self.ft = fasttext.load_model(model)
        
    def Vectrizer(self, wordlist):
        #平均ベクトル出力
        veclists = [] # 単語ベクトル出力
        for word in wordlist:
            W = self.ft[word]
            veclists.append(np.array(W,dtype='float16'))
        doc_vec=np.array(veclists).mean(axis=0)
        
        return   (doc_vec)   

#sBERTでモデル作成して保存
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path,)
                                                              
        self.model = BertModel.from_pretrained(model_name_or_path)
                                              
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings).numpy()



#カラムをリストへ
def FOR_LIST(df):
        LIST=df.values.tolist()
        LIST=[str(x)for x in LIST]
        LIST=[x for x in LIST if x != 'nan']
        return LIST



###
section_dic={  "脳内":"neurology", 
               "内泌": "endocrinology",
               "救急": "emergency",
               "心外": "cardiovascular_surgery",
               "脳外": "neurosurgery",
               "消内": "gastroenterology",
               "整形": "orthopedic surgery",
               "泌尿": "urology",
               "小児": "pediatrics",
               "消外": "Gastrointestinal surgery",
               "循内": "cardiovascular medicine",
               "皮膚": "dermatology",
               "母女": "obstetrics and gynecology",
               "呼内": "Respiratory Medicine",
               "総診": "emergency",
               "放科": "roentgenology",
               "耳鼻": "otorhinolaryngology",
               "ﾍﾟｲﾝ": "Anesthesiology",
               "呼外": "respiratory surgery",
               "乳一": "breast surgery",
               "腎内": "nephrology",
               "腫内": "oncology",
               "血内": "hematology",
               "眼科": "ophthalmology",
               "精神": "Psychiatry",
               "口外": "dental surgery",
               "形外": "plastic surgery",
  }    

def rename_section(df):#不要
    df.rename(columns=section_dic, inplace=True)
    return df

def CSVfordf(csv):
    df = pd.read_csv(csv)
    # 半角、スペース、小文字修正
    df['diagnosis'] = df['diagnosis'].map(mojimoji.zen_to_han)
    df['purpose'] = df['purpose'].map(mojimoji.zen_to_han)
    df['diagnosis']=df['diagnosis'].str.replace(' ', '')
    df['purpose']=df['purpose'].str.replace(' ', '')
    df['diagnosis']=df['diagnosis'].str.lower()
    df['purpose']=df['purpose'].str.lower()
    df['new_diagnosis'] = df['diagnosis'].copy().apply(meishi)
    # 重複をなくす
    df['new_diagnosis']=df['new_diagnosis'].map(set)
    df['new_diagnosis']=df['new_diagnosis'].map(list)

    # fasttextをインスタンス化
    from modules import FastText_Vectrizer
    FT=FastText_Vectrizer("../data/model/fasttext_meishi_model_100.bin")
    Tovec = FT.Vectrizer
    
    # new_diagnosis（名詞群）をベクトル化し平均をdfに追加
    df_vec = df['new_diagnosis'].apply(Tovec)
    
    # カラム名を変更する
    df_vec=list(df_vec)
    num=df_vec[0].shape[0]
    col_name = ["Dvec"+str(i) for i in range(num)]
    df_vec=pd.DataFrame(df_vec,columns=col_name)
    df = pd.concat([df,df_vec],axis=1)
    
    return df


def exam_preprosses(df):#不要
    # sectionの主成分分析モデルの作成
    pca = PCA(n_components=0.9)
    pca.fit(df_section)
    pca_comp = np.asarray(pca.components_)
    np.save('pca_comp_section', pca_comp)
    section_values = pca.transform(df_section)

    num=section_values[0].shape[0]
    col_name = ["section_pca"+str(i) for i in range(num)]
    section_pca=pd.DataFrame(section_values,columns=col_name)
    
    df_position =one_hot(df, 'position')
    # positionの主成分分析モデルの作成
    pca = PCA(n_components=0.9)
    pca.fit(df_position)
    pca_comp = np.asarray(pca.components_)
    np.save('pca_comp_position', pca_comp)
    position_values = pca.transform(df_position)

    num=position_values[0].shape[0]
    col_name = ["position_pca"+str(i) for i in range(num)]
    position_pca=pd.DataFrame(position_values,columns=col_name)
    
    df = pd.concat([df, section_pca,position_pca], axis=1)

    return df

#sentencebert
def sBERT_model(path):
    # sBERTファインチューニングしたのモデル
    MODEL_NAME = "../data/model/strf_sonoisa_sentence-bert-base-ja-mean-tokens-v232.75.10"
    word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=75)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 学習データを上のモデルでベクトル変換したデータ
    train_embeddings=np.load('../data/exam_data/sonoisa_vec/sbvec_so.npy')

    # 主成分分析モデルの作成
    pca = PCA(n_components=0.9)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)
    #pcaのパラメータ保存
    np.save('pca_comp_BERT', pca_comp)
    #pca_comp=np.load('pca_comp.npy')
    # 主成分分析モデルをBERTの最後に足す
    new_dimension=139    
    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), 
                         out_features=new_dimension, bias=False, 
                         activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)
    
    return model

#modelの保存
#pd.to_pickle(model, 'smodel.pkl')
def use_sBERT_model(df, model, path):
    #カラム名変更
    sBERT = model.encode(df["purpose"])
    sBERT=list(sBERT)
    num=sBERT[0].shape[0]
    col_name = ["P(S)vec"+str(i) for i in range(num)]
    sBERT=pd.DataFrame(sBERT,columns=col_name)
    df_sBERT = pd.concat([df,sBERT],axis=1)
    df_sBERT.to_csv(path)    
    
    return df_sBERT


def use_fasttext_model(df, model, path):
        # fasttextをインスタンス化
    from modules import FastText_Vectrizer
    FT=FastText_Vectrizer(model)
    Tovec = FT.Vectrizer
    # new_diagnosis（名詞群）をベクトル化し平均をdfに追加
    df_vec = df['purpose'].apply(Tovec)
    
        # カラム名を変更する
    df_vec=list(df_vec)
    num=df_vec[0].shape[0]
    col_name = ["P(f)vec"+str(i) for i in range(num)]
    df_vec=pd.DataFrame(df_vec,columns=col_name)
    df_ft = pd.concat([df,df_vec],axis=1)
    
        # ベクトルデータへ変換後のデータを保存
    df_ft.to_csv("CSVs/ft_data.csv")
    
    return df_ft

def label_encording(df):
    from sklearn import preprocessing
    lbl_s = preprocessing.LabelEncoder()
    lbl_s.fit(df['section'])
    lbl_section = lbl_s.transform(df['section'])

    lbl_p = preprocessing.LabelEncoder()
    lbl_p.fit(df['position'])
    lbl_position = lbl_p.transform(df['position'])

    lbl_l = preprocessing.LabelEncoder()
    lbl_l.fit(df['label'])
    lbl_label = lbl_l.transform(df['label'])
    
    y = lbl_label
    X = df.drop([ 'label', 'section', 'position', 'new_diagnosis'], axis=1)
    X['sec_lbl'] = lbl_section
    X['pos_lbl'] = lbl_position

    return X ,y

def Add_class_wight(X, y):
    class_weights = list(class_weight.compute_class_weight('balanced', 
                                                           classes=np.unique(y),
                                                           y=y)
                        )
    w_array = np.ones(y.shape[0], dtype = 'float16')
    for i, val in enumerate(y):
        w_array[i] = class_weights[val]
    
    
    
    return w_array


############################# LBGM


def run_LGBM(X_train_cv, y_train_cv,X_eval_cv, y_eval_cv ,CF,CW):
    # 学習用
    lgb_train = lgb.Dataset(X_train_cv, y_train_cv,
                            categorical_feature=CF,
                            free_raw_data=False,
                            weight=CW)
    # 検証用
    lgb_eval = lgb.Dataset(X_eval_cv, y_eval_cv, reference=lgb_train,
                           categorical_feature=CF,
                           free_raw_data=False,
                           weight=np.ones(len(X_eval_cv)).astype('float16'))
    
    # パラメータを設定
    params = {'task': 'train',                # 学習、トレーニング ⇔　予測predict
              'boosting_type': 'gbdt',        # 勾配ブースティング
              'objective': 'multiclass',      # 目的関数：多値分類、マルチクラス分類
              'metric': 'multi_logloss',      # 分類モデルの性能を測る指標
              'num_class': 55,                 # 目的変数のクラス数
              'learning_rate': 0.05,          # 学習率（初期値0.1）
              'num_leaves': 23,               # 決定木の複雑度を調整（初期値31）
              'min_data_in_leaf': 1,          # データの最小数（初期値20）
             }
    
    # 学習
    evaluation_results = {}                                     # 学習の経過を保存する箱
    model = lgb.train(params,                                   # 上記で設定したパラメータ
                      lgb_train,                                # 使用するデータセット
                      num_boost_round=1000,                     # 学習の回数
                      valid_names=['train', 'valid'],           # 学習経過で表示する名称
                      valid_sets=[lgb_train, lgb_eval],         # モデル検証のデータセット
                      evals_result=evaluation_results,          # 学習の経過を保存
                      categorical_feature=CF, # カテゴリー変数を設定
                      early_stopping_rounds=20,                 # アーリーストッピング
                      verbose_eval=-1)                          # 学習の経過の非表示
    
    return model, evaluation_results


def FI_LGBM(models, X):
    Is = []

    for i in range(len(models)):
        # feature importanceを表示
        importance = pd.DataFrame(models[i].feature_importance(), index=X.columns, columns=['importance'])
        Is.append(importance)
 
    return Is

###################################  NN
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

def make_nn_model(x_train, class_num):
    
    input_num = Input(shape=(x_train.shape[1],))
    x_num = Dense(200, activation='relu')(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)
    
    x_num = Dense(200, activation='relu')(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)
    
    x_num = Dense(200, activation='relu')(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)
    
    x_num = Dense(100, activation='relu')(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)
    
    x_num = Dense(100, activation='relu')(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.1)(x_num)
    
    out = Dense(class_num, activation='softmax')(x_num)
    
    model = Model(inputs=input_num, outputs=out)
  
    model.compile(
        optimizer=Adam(learning_rate=1e-2),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
        )

    return model



def make_results(y_pred, y_test):
    y_pred_max = np.argmax(y_pred, axis=1)
    df_result = pd.DataFrame(list(zip(y_test, y_pred_max)), columns = ['true','pred'])
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    df_report = pd.DataFrame(classification_report(y_pred_max, y_test, 
                                               output_dict=True)).T
    
    sns.heatmap(confusion_matrix(df_result['true'],df_result['pred']), annot=True)
    plt.xlabel("pred")
    plt.ylabel('true')
    
    return (df_result, df_report)

def model_evaluation(models, X_test, y_test, method='LBGM'):
    results = []
    reports = []
    
    for i in range(len(models)):
        if method=='LGBM':
            y_pred = models[i].predict(X_test, num_iteration=models[i].best_iteration)
        
        else :
            y_pred = models[i].predict(X_test)
            
        result, report = make_results(y_pred, y_test)
        results.append(result)
        reports.append(report)
        
    return results, reports