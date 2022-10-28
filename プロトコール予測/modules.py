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








def is_japanise(str):
    import unicodedata
    
    for ch in str:
        name = unicodedata.name(ch)
        
        if'CJK UNIFIED'in name or'HIRAGANA'in name or'KATAKANA'in name or'BLACK'in name or 'DIGIT' in name or 'SQUARE' in name:
            return True
        else:
            return False
#wakatigaki        
def wakati(Str):
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
    stop_words= [',','｡','.','右','左','両側','*','(',')','委任',':','。','、',',','.','+','疑い']
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


#カラムをリストへ
def FOR_LIST(df):
        LIST=df.values.tolist()
        LIST=[str(x)for x in LIST]
        LIST=[x for x in LIST if x != 'nan']
        return LIST

def TrainData(PATH):
        import pandas as pd
        import datetime
        from sklearn.preprocessing import OneHotEncoder
        import mojimoji


        df=pd.read_csv(PATH,encoding='ms932') 
        df['diagnosis']=df['diagnosis'].map(list(set()))
        
        df['diagnosis'] = df['diagnosis'].map(mojimoji.zen_to_han)
        df['purpose'] = df['purpose'].map(mojimoji.zen_to_han)
        df['diagnosis']=df['diagnosis'].str.replace(' ', '')
        df['purpose']=df['purpose'].str.replace(' ', '')
        df['diagnosis']=df['diagnosis'].str.lower()
        df['purpose']=df['purpose'].str.lower()
       
        
        
        
        df.loc[df['section'].str.contains('総診'), 'section'] = '救急'
        df.loc[df['section'].str.contains('ﾍﾟｲﾝ'), 'section'] = '麻酔'
        
        
        return df

def     one_hot(df, col):
        categories = df[col].unique()
        print(categories)
        df[col] = pd.Categorical(df[col], categories=categories)
        df_c = pd.get_dummies(df[col])
        df = pd.concat([df, df_c], axis=1)
        del df[col]
    
        return df


###

def position_transform(df):
        a = df["position"].unique()
        #各部位を新たな部位ごとにリストにする
        head = list(filter(lambda x: '頭部' in x,a))
        brain = list(filter(lambda x: '脳' in x,a))
        inn = list(filter(lambda x: '内耳(後頭蓋窩)' in x,a))
        pitu = list(filter(lambda x: '下垂体' in x,a))
        me  = list(filter(lambda x: '眼窩' in x,a))
        td  = list(filter(lambda x: 'MRdevice頭頸部' in x,a))
        sinus = list(filter(lambda x: '副鼻腔' in x,a))
        gaku = list(filter(lambda x: '顎' in x,a))
        
        
        head = head + brain + inn +pitu +me +td + sinus +gaku

        kasik = list(filter(lambda x: '下肢血管' in x,a))

        zyoushik = list(filter(lambda x: '上肢血管' in x,a))

        abd = list(filter(lambda x: '上腹部' in x,a))
        mrcp = list(filter(lambda x: '胆道' in x,a))
        ht = list(filter(lambda x: 'MRdevice腹部' in x,a))

        abd =abd + mrcp +ht

        uabd = list(filter(lambda x: '下腹部' in x,a))
        uro = list(filter(lambda x: '腎' in x,a))
        pro = list(filter(lambda x: '前立腺' in x,a))
        u = list(filter(lambda x: '膀胱' in x,a))
        pel = list(filter(lambda x: '骨盤' in x,a))
        uro2 = list(filter(lambda x: '尿路' in x,a))
        tyou = list(filter(lambda x: '腸管(経口法)' in x,a))

        uabd = uabd + pel + uro + pro + u + pel +uro2 +tyou

        spine = list(filter(lambda x: '全脊椎' in x,a))
        spine2 = list(filter(lambda x: '椎' in x,a))
        spine =spine +spine2

        neck = list(filter(lambda x: '頚部' in x,a))

        ch = list(filter(lambda x: '胸' in x,a))
        ch2 = list(filter(lambda x: '縦隔' in x,a))
        ch =ch +ch2

        man = list(filter(lambda x: '乳' in x,a))


        afg = list(filter(lambda x: '胎児' in x,a))

        hand = list(filter(lambda x: '手' in x,a))
        foot = list(filter(lambda x: '足' in x,a))

        zs = list(filter(lambda x: '上肢' in x,a))
        hiji = list(filter(lambda x: '肘' in x,a))
        ude = list(filter(lambda x: '上腕' in x,a))
        zen = list(filter(lambda x: '前腕' in x,a))
        kata = list(filter(lambda x: '肩' in x,a))

        zs = zs + hiji + ude +zen +kata

        ks = list(filter(lambda x: '下肢' in x,a))
        katai = list(filter(lambda x: '下腿' in x,a))
        hip = list(filter(lambda x: '股関節' in x,a))
        hemo = list(filter(lambda x: '大腿部' in x,a))
        knee = list(filter(lambda x: '膝' in x,a))

        ks = ks + katai + hip +hemo + knee
        
        heart = list(filter(lambda x: '心臓' in x,a))
        

        #部位名の変更
        df['position']=df["position"].replace(head, 'brain')
        df['position']=df["position"].replace(kasik, 'blood vessel(leg)')
        df['position']=df["position"].replace(zyoushik, 'blood vessel(upper)')
        df['position']=df["position"].replace(abd, 'abdomen')
        df['position']=df["position"].replace(uabd, 'pelvis')
        df['position']=df["position"].replace(spine, 'spine')
        df['position']=df["position"].replace(neck, 'neck')
        df['position']=df["position"].replace(ch, 'chest')
        df['position']=df["position"].replace(man, 'breast')
        df['position']=df["position"].replace(afg, 'fetus')
        df['position']=df["position"].replace(hand, 'hand')
        df['position']=df["position"].replace(foot, 'foot')
        df['position']=df["position"].replace(zs, 'upper_limb')
        df['position']=df["position"].replace(ks, 'lower_limb')
        df['position']=df["position"].replace(heart, 'heart')
        

        return df
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
    df=rename_section(df)
    
    df_section =one_hot(df,'section')
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



def to_vec(df):
    FT=FastText_Vectrizer("model/fasttext_meishi_model_100.bin")
    Tovec = FT.Vectrizer
    
    MODEL_NAME = "model/strf_sonoisa_sentence-bert-base-ja-mean-tokens-v232.75.10"
    word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=75)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_embeddings=np.load('exam_data/sonoisa_vec/sbvec_so.npy')
    # 主成分分析モデルの作成
    pca = PCA(n_components=0.9)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)
    #pcaのパラメータ保存
    np.save('pca_comp_BERT', pca_comp)
    #pca_comp=np.load('pca_comp.npy')
    # 主成分分析モデルをBERTの最後に足す
    new_dimension=139    
    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)
    
    df['new_diagnosis'] = df['diagnosis'].copy().apply(meishi)
    df_vec = df['new_diagnosis'].apply(Tovec)
    df_vec=list(df_vec)
    num=df_vec[0].shape[0]
    col_name = ["diag_vec"+str(i) for i in range(num)]
    df_vec=pd.DataFrame(df_vec,columns=col_name)
    df = pd.concat([df,df_vec],axis=1)
    
    sBERT = model.encode(df["purpose"])
    sBERT=list(sBERT)
    num=sBERT[0].shape[0]
    col_name = ["pur_vec"+str(i) for i in range(num)]
    sBERT=pd.DataFrame(sBERT,columns=col_name)
    df = pd.concat([df,sBERT],axis=1)
    
    return df
