import warnings
warnings.filterwarnings("ignore")

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import nltk
from pyltp import SentenceSplitter
from tqdm import tqdm


class Translator():
    def __init__(self, model_path, use_gpu):
        self.use_gpu = use_gpu
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        if use_gpu:
            self.model = self.model.cuda()
        self.model = self.model.eval()
        
    def translate(self, text, source, target):
        self.tokenizer.src_lang = source
        encoded_zh = self.tokenizer(text, return_tensors="pt")
        if self.use_gpu:
            encoded_zh['input_ids'] = encoded_zh['input_ids'].cuda()
            encoded_zh['attention_mask'] = encoded_zh['attention_mask'].cuda()
        generated_tokens = self.model.generate(**encoded_zh, forced_bos_token_id=self.tokenizer.get_lang_id(target))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    def translate_zh_en(self, text):
        sentences = SentenceSplitter.split(text)
        res = []
        for sentence in tqdm(sentences):
            res.append(self.translate(sentence, "zh", "en"))
        return "".join(res)
            
    
    def translate_en_zh(self, text):
        sentences = nltk.sent_tokenize(text)
        res = []
        for sentence in tqdm(sentences):
            res.append(self.translate(sentence, "en", "zh"))
        return "".join(res)