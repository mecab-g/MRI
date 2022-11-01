#日本語が混じっている単語のリスト作成
import re
from transformers import BertJapaneseTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import fasttext 
from sklearn.decomposition import PCA
from sentence_transformers import models, losses, evaluation, SentenceTransformer

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
def makeAug(df, List):
    df2 = df.copy()
    #df.loc[:,'purpose']= List1
    #df2.loc[:,'purpose']= List2
    num=int(df2['year'].mean())
    generator = np.random.default_rng()
    rnd = generator.normal(loc=num, scale=10, size=len(df))
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

def rename_section(df):
    df.rename(columns=section_dic, inplace=True)
    return df

def exam_preprosses(df):
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




