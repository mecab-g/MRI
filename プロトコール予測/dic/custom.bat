echo on

cd "C:\Program Files\MeCab\bin" 
mecab-dict-index.exe -d "C:/Program Files/MeCab/dic/ipadic" -u "C:\Users\yu\Desktop\MRIdic.dic" -f sjis -t utf-8 "C:\Program Files\MeCab\dic\userdic\MRIdic.csv"

pause