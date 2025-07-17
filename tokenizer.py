from modules import *

"""
Code được viết hoàn toàn bởi phú.
Hoàn thành 2/7/2025
Colab
Dự án cá nhân
https://colab.research.google.com/drive/1Ou7DE-77I5DKDowbY6vzyG5kq5dggXiW#scrollTo=jnOcoSwbysVT
"""

def ijson_loader(path):
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for item in ijson.items(f, "item"):
            result.append(item)
    return result

class TokenizeTrainer:
    def __init__(self,
                 data_dir="./tokenizer",
                 pad_token="[pad]",
                 unk_token="[unk]",
                 start_token="[start]",
                 eos_token="[eos]"):

        self.data_dir = data_dir
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.eos_token = eos_token

        self.data = list()
        self.tokenize = list()
        self.cache = self.cache_init()
        self.vocab = dict()
        self.dictionary = set()



    # khởi tạo các file/folder cần thiết nếu chưa tồn tại
    def init_folder_file(self, vocab_path, cache_path, tokenize_path):
        # khởi tạo folder
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # ghép các file path vào path folder
        full_vocab_path = os.path.join(self.data_dir, vocab_path)
        full_cache_path = os.path.join(self.data_dir, cache_path)
        full_tokenize_path = os.path.join(self.data_dir, tokenize_path)

        # khởi tạo các files nếu các files chưa tồn tại
        if not os.path.exists(full_vocab_path):
            with open(full_vocab_path, "w", encoding="utf-8"): pass
        if not os.path.exists(full_cache_path):
            with open(full_cache_path, "w", encoding="utf-8"): pass
        if not os.path.exists(full_tokenize_path):
            with open(full_tokenize_path, "w", encoding="utf-8"): pass



    # xử lý chuyển đổi batch từng câu thành danh dánh tokenize có dạng [['', ''], ['', ''],]
    def text_tokenize(self, data, space_process=True):
        return [ # xử lý khoảng cách nếu space_process
            list(word + " ") if space_process and not word.endswith("<NOTSPACE>") else (list(word.replace("<NOTSPACE>", "")) if space_process else list(word))
            for word in (" ".join([text + "<NOTSPACE> " for text in data]).split() if space_process else " ".join(data).split())
        ] + (
            [[" "]] + [['\n'], ['\r'], ['\t'], ['\b'], ['\f'], ['\v'], ['\0']] \
            + [[char] for char in "~`!@#$%^&*()_+-={}|[]\\:\";'<>?,./"]
        )



    # nạp data vào, data có dạng danh sách các câu
    def fit(self, data):
        self.data = data # data đã nạp

        # khởi tạo các special token vào vocab
        self.vocab[self.pad_token] = 0
        self.vocab[self.unk_token] = 1
        self.vocab[self.start_token] = 2
        self.vocab[self.eos_token] = 3

        # nạp các danh sách ký tự cơ bản lấy từ batch data vào vocab
        for char in set(" ".join(data)):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # chuyển batch data thành danh sách tokenize
        self.tokenize = self.text_tokenize(data, True)

        # tạo bộ từ điển để sau này có thể cần tra cứu
        data = set(" ".join(data).split())
        for word in data:
            if len(word) >= 2 and word not in self.dictionary:
                self.dictionary.add(word)



    # lưu các files
    def save(self, vocab_path="vocab.json", cache_path="cache.json", tokenize_path="tokenize.json", dictionary_path="dictionary.json", save_cache=False, save_tokenize=True, save_dictionary=True, verbose=True):
        self.init_folder_file(vocab_path, cache_path, tokenize_path)
        reverse_vocab = {str(v): k for k, v in self.vocab.items()} # đảo ngược lại vocab

        def save_voccab_thread():
            # lưu dict từ vựng với 2 dạng, vocab to index (từ vựng sang chỉ số) và index to vocab (chỉ số sang từ vựng)
            with open(os.path.join(self.data_dir, vocab_path), "w", encoding="utf-8") as f:
                json.dump({"voc_to_idx": self.vocab, "idx_to_voc": reverse_vocab}, f, indent=4, ensure_ascii=False)

        def save_cache_thread():
            with open(os.path.join(self.data_dir, cache_path), "w", encoding="utf-8") as f:
                hist_token_freq_computed, path_reverse, counter, paths = self.cache
                obj = {
                    "hist_token_freq_computed": list(hist_token_freq_computed),
                    "path_reverse": path_reverse,
                    "counter": {"<NOTMERGE>".join(k): v for k, v in counter.items()}
                }
                json.dump(obj, f, indent=4, ensure_ascii=False)

        def save_tokenize_thread():
            with open(os.path.join(self.data_dir, tokenize_path), "w", encoding="utf-8") as f:
                json.dump(self.tokenize, f, indent=4, ensure_ascii=False)

        def save_dictionary_thread():
            with open(os.path.join(self.data_dir, dictionary_path), "w", encoding="utf-8") as f:
                json.dump(list(self.dictionary), f, indent=4, ensure_ascii=False)

        t_vocab_save = threading.Thread(target=save_voccab_thread)
        t_cache_save = threading.Thread(target=save_cache_thread)
        t_tokenize_save = threading.Thread(target=save_tokenize_thread)
        t_dictionary_save = threading.Thread(target=save_dictionary_thread)

        t_vocab_save.start()
        # lưu cache
        if save_cache:
            t_cache_save.start()
        # lưu tokenize
        if save_tokenize:
            t_tokenize_save.start()
        # lưu dictionary
        if save_dictionary:
            t_dictionary_save.start()

        if verbose:
            print("Vocab saving")
        t_vocab_save.join()
        if save_cache:
            if verbose:
                print("Cache saving")
            t_cache_save.join()
        if save_tokenize:
            if verbose:
                print("Tokenize saving")
            t_tokenize_save.join()
        if save_dictionary:
            if verbose:
                print("Dictionary saving")
            t_dictionary_save.join()



    def load(self, vocab_path="vocab.json", cache_path="cache.json", tokenize_path="tokenize.json", dictionary_path="dictionary.json", load_cache=False, load_tokenize=True, load_dictionary=True, verbose=True):
        def load_vocab_thread():
            with open(os.path.join(self.data_dir, vocab_path), "r", encoding="utf-8") as f:
                vocab = json.load(f)
                self.vocab = vocab["voc_to_idx"]

        def load_cache_thread():
            with open(os.path.join(self.data_dir, cache_path), "r", encoding="utf-8") as f:
                cache = json.load(f)
                self.cache = (
                    set(tuple(x) for x in cache["hist_token_freq_computed"]),
                    cache['path_reverse'],
                    Counter({tuple(k.split("<NOTMERGE>")): v for k,v in cache["counter"].items()}),
                    []
                )

        def load_tokenize_thread():
            self.tokenize = ijson_loader(os.path.join(self.data_dir, tokenize_path))

        def load_dictionary_thread():
            self.dictionary = set(ijson_loader(os.path.join(self.data_dir, dictionary_path)))

        t_vocab_load = threading.Thread(target=load_vocab_thread)
        t_cache_load = threading.Thread(target=load_cache_thread)
        t_tokenize_load = threading.Thread(target=load_tokenize_thread)
        t_dictionary_load = threading.Thread(target=load_dictionary_thread)

        if load_cache:
            t_cache_load.start()
        if load_tokenize:
            t_tokenize_load.start()
        if load_dictionary:
            t_dictionary_load.start()

        t_vocab_load.start()

        if load_cache:
            if verbose:
                print("Cache loading")
            t_cache_load.join()
        if load_tokenize:
            if verbose:
                print("Tokenize loading")
            t_tokenize_load.join()
        if load_dictionary:
            if verbose:
                print("Dictionary loading")
            t_dictionary_load.join()
        if verbose:
            print("Vocab loading")
        t_vocab_load.join()



    # khởi tạo cache cho quá trình merge (cache khiến skip các xử lý dư thừa, giúp tăng tốc)
    def cache_init(self):
        """
        hist_token_freq_compute: Đây là cache giúp vòng lặp tính merge freq skip đi các cặp merge đã tính trước đó
        path_reverse: Đây là cache dạng dict, chứa keys là các cặp merge trong quá trình merge, và value là các đường dẫn
        đến các từ vựng cần thay thế trong self.tokenize
        paths: Đây là cache đã index ra từ path_reverse key index là cặp từ phổ biến nhất
        pair_counter: Đây là cache đếm số lần merge được thực hiện
        """
        hist_token_freq_computed = set()
        path_reverse = dict()
        pair_counter = Counter()
        paths = []
        return hist_token_freq_computed, path_reverse, pair_counter, paths


    # tính toán tần suất của các cặp từ con liền kỳ
    def merge_freq(self, tokenize, cache):
        # unpack các cache ra ngoài
        hist_token_freq_computed_global, path_reverse, counter, paths = cache

        pair_list, hist_token_computed_local = [], set() # danh sách chứa các cặp từ đã merge (sau này ép sang Counter để lấy tần suất)
        for i in range(len(tokenize)): # lặp index từng từ
            # skip từ đã merge trước đó giúp tăng tốc độ và lưu lại từ đã tính hiện tại, để tránh tính lần sau
            if tuple(tokenize[i]) in hist_token_freq_computed_global:
                continue
            hist_token_computed_local.add(tuple(tokenize[i]))

            # merge các cặp từ
            for j in range(len(tokenize[i]) - 1):

                # skip các pair đã tính
                if (tokenize[i][j], tokenize[i][j+1]) in counter:
                    continue

                pair = (tokenize[i][j], tokenize[i][j+1])
                pair_list.append(pair)

                if str(pair) not in path_reverse:
                    path_reverse[str(pair)] = [i]
                else:
                    path_reverse[str(pair)].append(i)

        # ép danh sách đã merge (pair_list) vào counter để counter update tuần suất cho các cặp từ con mới
        counter.update(pair_list)
        most_common_pair, freq = counter.most_common(1)[0] # lấy cặp từ phổ biến nhất
        counter.pop(most_common_pair) # xóa cặp từ phổ biến khỏi counter
        paths = path_reverse[str(most_common_pair)] # lấy đường dẫn cần update của các từ có chứa subword phổ biến nhất để replace ở hàm update_merge

        # packing lại cache
        hist_token_freq_computed_global |= hist_token_computed_local # kết hợp 2 set cache lại
        cache = (hist_token_freq_computed_global, path_reverse, counter, paths)

        return most_common_pair, freq, cache



    # thay thế các tokenize mới bằng các tokenize đã ghép
    def update_merge(self, tokenize, most_common_pair, cache):
        hist_token_freq_computed, path_reverse, counter, paths = cache # unpacke cache
        hist_token_for_avoid_repeat = {} # cache local tránh thực hiện việc ghép cặp, giúp tăng tốc

        for idx in paths: # vòng lặp truy xuất các index trong self.tokenize cần thay thế
            word = tokenize[idx] # lấy token (word) trong self.tokenize dựa trên index đã lưu
            word_text = "".join(word) # ghép token lại thành một từ đơn dạng text

            if word_text not in hist_token_for_avoid_repeat: # nếu từ chưa có trong cache sẽ thực hiện xử lý ghép cặp và thay thế
                # thuật toán ghép cặp
                len_word = len(word)
                i = 0
                new_token = list()

                while i < len_word:
                    if i == len_word - 1:
                        new_token.append(word[i])
                        break
                    if (word[i], word[i+1]) == most_common_pair:
                        new_token.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_token.append(word[i])
                        i += 1
                # thay thế token đã ghép và lưu lại token vào cache
                tokenize[idx] = new_token
                hist_token_for_avoid_repeat[word_text] = new_token

            else: # truy xuất token đã lưu trong cache trực tiếp tránh tính toán lại
                tokenize[idx] = hist_token_for_avoid_repeat[word_text]

        cache = (hist_token_freq_computed, path_reverse, counter, paths)
        return tokenize, cache



    def train(self, epochs=1000):
        for i in range(epochs):
            most_common_pair, freq, self.cache = self.merge_freq(self.tokenize, self.cache)
            if most_common_pair[0] + most_common_pair[1] not in self.vocab:
                self.vocab[most_common_pair[0] + most_common_pair[1]] = len(self.vocab)
            print(f"\rKỷ nguyên: {i+1}, Cặp từ con: {most_common_pair}, Tần suất: {freq}", end="", flush=True)
            if freq == 1:
                print("Không còn từ vựng để học")
                break
            self.tokenize, self.cache = self.update_merge(self.tokenize, most_common_pair, self.cache)
            if i == epochs - 1:
                print()


class Tokenizer(TokenizeTrainer):
    def __init__(self, data_dir="./tokenizer", smart_split_word_in_noise_word=True):
        super().__init__(data_dir)
        self.data_dir = data_dir
        self.encode_cache = dict()
        self.vocab = dict()
        self.reverse_vocab = dict()
        self.vocab_size = 0
        self.smart_split_word_in_noise_word = smart_split_word_in_noise_word

        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.start_token_id = None

    def load(self, vocab_path="vocab.json", dictionary_path="dictionary.json"):
        vocab = json.load(open(os.path.join(self.data_dir, vocab_path), "r", encoding="utf-8"))
        self.vocab = vocab["voc_to_idx"]
        self.reverse_vocab = vocab["idx_to_voc"]
        if self.smart_split_word_in_noise_word:
            self.dictionary = set(json.load(open(os.path.join(self.data_dir, dictionary_path), "r", encoding="utf-8")))
        self.vocab_size = len(self.vocab)
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.start_token_id = self.vocab[self.start_token]


    def split_known_word(self, sent, dictionary):
        i, j = 0, len(sent)
        special_characters = ["\n", "\r", "\t", "\b", "\f", "\v", "\0"]
        new_sent = ""

        while i < len(sent):

            sub = sent[i: j]

            if sub in dictionary:
                i = j
                j = len(sent)
                new_sent += " " + sub + " "

            elif j == (i + 1):
                i += 1
                j = len(sent)
                new_sent += sub
                continue

            else:
                j -= 1

        return new_sent.strip(" ").replace("  ", " ").replace("  ", " ")


    def encode(self, text: str, return_tensor_type="list", maxlen_word=20): # return_tensor_type: [list or tensor]
        sent_tokenize = list()

        text = text.replace(" ", "<THISISSPACE> ")
        text_split = text.split(" ")
        text_space_process = [
            text_split[i].replace("<THISISSPACE>", " ") for i in range(len(text_split))
        ]

        punctions = list("~`!@#$%^&*()_+-={}|[]\\:\";'<>?,./")
        special_characters = ["\n", "\r", "\t", "\b", "\f", "\v", "\0"]

        for word_first in text_space_process:
            # tokenize word từ cache giúp tăng tốc
            if word_first in self.encode_cache:
                sent_tokenize += self.encode_cache[word_first]
                continue

            origin_word = word_first
            word_tokenize = []

            for word in (self.split_known_word(word_first, self.dictionary).split(" ") if self.smart_split_word_in_noise_word else [word_first]):
                word = word[:maxlen_word] # giảm length giúp tránh các từ quá dài, ảnh hưởng encode

                if word == "":
                    continue
                # xử lý dấu câu tránh encode word dính liền dấu
                for p in punctions:
                    word = word.replace(p, f" {p}")

                if word[-1] != " ":
                    word += " "
                # xử lý ký tự đặc biệt ở cuối
                if word[-1] in special_characters:
                    for c in special_characters:
                        word = word.replace(c, f" {c}")
                    if word[-1] == " ":
                        word.strip(" ")

                # nhất quán giữa token có space và token không space tránh tạo ra 2 biểu diễn cho cùng loại token chỉ vì space
                elif " " != word[-1]:
                    word += " "

                # tránh thêm dấu cách do mã phía trên, nếu từ đơn là ký tự đặc biệt
                if word.strip(" ") in special_characters:
                    word = word.strip(" ")

                # tránh thêm dấu cách do mã phía trên, nếu từ đơn là dấu câu
                if word.strip(" ") in punctions:
                    word = word.strip(" ")

                # scan từ để tạo sentence tokenize
                i, j = 0, len(word)
                space_vocab_loss_slot_flag = False
                while i < len(word):
                    if j <= i:
                        word_tokenize.append(self.vocab[self.unk_token])
                        i += 1
                        j = len(word)
                        continue

                    # xử lý scan vocab với space
                    if word[i: j] in self.vocab:
                        word_tokenize.append(self.vocab[word[i: j]])
                        i = j
                        j = len(word)
                    else:
                        space_vocab_loss_slot_flag = True
                        j -= 1

            sent_tokenize += word_tokenize
            self.encode_cache[origin_word] = word_tokenize

        # loại bỏ dấu space ở cuối do quá trình xử lý, tránh thêm token không cần thiết
        if sent_tokenize[-1] == self.vocab[" "]:
            sent_tokenize = sent_tokenize[:-1]

        return sent_tokenize if return_tensor_type == "list" else torch.tensor(sent_tokenize)


    def decode(self, sent_tokenize, skip_special_token=True):
        text_decode = str()
        punctions = list("~`!@#$%^&*()_+-={}|[]\\:\";'<>?,./")
        for token in sent_tokenize:
            new_word = self.reverse_vocab[str(int(token))]
            text_decode += new_word
        # xử lý punction
        for p in punctions:
            if p in text_decode:
                text_decode = text_decode.replace(f" {p}", f"{p}")
        # bỏ qua token đặc biệt khi decode
        if skip_special_token:
            for token in [self.eos_token, self.pad_token, self.unk_token, self.start_token]:
                text_decode = text_decode.replace(f"{token}", "")
        return text_decode.strip(" ")

    def batch_encode(self, batch_text, padding=True, return_tensor_type="list"):
        output = [self.encode(text, "list") for text in batch_text]
        attention_mask = []

        if padding:
            max_len = max([len(sent_tokenize) for sent_tokenize in output])
            for i in range(len(output)):
                output_i_length = len(output[i])
                attention_mask.append( [1] * output_i_length + [0] * (max_len - output_i_length))
                output[i] += [self.pad_token_id] * (max_len - output_i_length)
        else:
            min_len = min([len(sent_tokenize) for sent_tokenize in output])
            for i in range(len(output)):
                attention_mask.append([1] * min_len)
                output[i] = output[i][:min_len]

        if return_tensor_type == "tensor":
            output = {"input_ids": torch.tensor(output), "attention_mask": torch.tensor(attention_mask)}
        else:
            output = {"input_ids": output, "attention_mask": attention_mask}

        return output

    def batch_decode(self, batch_sent_tokenize, skip_special_token=True):
        return [self.decode(sent_tokenize, skip_special_token) for sent_tokenize in batch_sent_tokenize]

    # contexts có dạng [{"role": "system hoặc assistant hoặc user", "content": "..."}]
    def apply_chat_template(self, contexts, train_template=False, return_text=False, return_tensor_type="list"):
        text = ""
        for context in contexts[:-1]:
            role = context["role"]
            content = context["content"]
            text += f"<|{role}|>{self.start_token}\n{content}{self.eos_token}\n"

        if train_template:
            text += f"<|{contexts[-1]['role']}|>{self.start_token}\n{contexts[-1]['content']}{self.eos_token}"
        else:
            text += f"<|{contexts[-1]['role']}|>{self.start_token}\n{contexts[-1]['content']}{self.eos_token}\n"
            text += f"<|{contexts[-2]['role']}|>{self.start_token}\n"

        if return_text:
            return text

        output = self.encode(text, "list")
        return torch.tensor(output) if return_tensor_type == "tensor" else output

    def batch_apply_chat_template(self, batch_contexts, train_template=False, padding=True, return_tensor_type="list"):
        return self.batch_encode(
            [self.apply_chat_template(contexts, train_template, return_text=True) for contexts in batch_contexts],
            return_tensor_type=return_tensor_type,
            padding=padding
        )
