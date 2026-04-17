import os,sys
import glob
import json

#    {
#        "id": "voice10_s1K382_emo5",
#        "audio": "sample_dataset/output/voice10_s1K382_emo5.wav",
#        "text": "我真的要小扁豆凝集素",
#        "speaker": "voice10"
#    },

# label.txt
#U00004  四月份貿易逆差為九千三百零四億日圓，優於市場預測的五千六百億日圓。
#        si4 yue4 fen4 mao4 yi4 ni4 cha1 wei2 jiu3 qian1 san1 bai3 ling2 si4 yi4 ri4 yuan2 , you1 yu2 shi4 chang3 yu4 ce4 de5 wu3 qian1 liu4 bai3 yi4 ri4 yuan2 .


results = []

# convert tradtional chinese characters into simplified version
def convert_to_simp(text):
    import opencc
    # 1. Basic OpenCC conversion (handles most characters)
    converter = opencc.OpenCC('tw2s')
    simplified = converter.convert(text)

    manual_map = {
            "吋": "寸", "呎": "尺", "妳": "你", "姵": "佩", "峇": "巴",
            "揹": "背", "暱": "昵", "牠": "它", "瓈": "璃", "瞇": "眯",
            "砲": "炮", "粧": "妆", "罣": "挂", "舖": "铺", "藷": "薯",
            "衞": "卫", "遶": "绕", "黐": "chi1", "徬": "彷", "暸": "liao2",
            "煇": "辉", "抬": "抬"}


    # 2. Apply our manual override for specific characters
    for trad, simp in manual_map.items():
        simplified = simplified.replace(trad, simp)

    return simplified

def read_labels(basedir,speaker):
    labelfile = basedir + "/"+speaker+"/Labeling/label.txt"
    wavdir = basedir + "/"+speaker+"/wavconverted/"
    record = {}

    with open(labelfile,"r",encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line[0]=="U":
                # chinese line
                sent_id = line[0:6]
                text_content = convert_to_simp(line[6:])
                text_content = text_content.strip()
                #sent_id,text_content = line.split("\t")
                text_id = speaker+"_"+sent_id
                wav_path = wavdir+sent_id+".wav"

                if os.path.isfile(wav_path):
                    record = {
                        "id": text_id,
                        "audio": wav_path,
                        "text": text_content,
                        "speaker": speaker,
                        "pinyin":"" }
                    #results.append(record)
            else: # pinyin sequence
                record["pinyin"] = line
                results.append(record)

    return results



def main():
    basedir = "/mnt/data/jia/index-tts/seasalt-tts-data/"
    spk1 = "tongtong"
    spk2 = "vivian"
    #basedir,spk1,spk2 = sys.argv[1],sys.argv[2],sys.argv[3]
    output_file = sys.argv[1]

    items1 = read_labels(basedir,spk1)
    items2 = read_labels(basedir,spk2)

    # Output to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Successfully processed {len(results)} files into {output_file}")


main()
