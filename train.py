from modules import *
from model import Causal
from tokenizer import Tokenizer

def load_json_list_safely(path):
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for item in ijson.items(f, "item"):
            result.append(item)
    return result

"""
Load data giữa các đợt và kết hợp lại với nhau, theo kế hoạch crawl, sẽ có 7 - 10 đợt dataset được crawl
mỗi đợt data chứa khoảng 100 nghìn câu, nguồn được crawl là tổng hợp nhiều nguồn website tiếng việt, như:
sách, báo chí, wiki, truyện, blog cá nhân, truyện người lớn, diễn đàn, và hội thoại chat..vv
"""

# load data
data1 = load_json_list_safely('./vi_dataset/data1.json')
data2 = load_json_list_safely('./vi_dataset/data2.json')
data3 = load_json_list_safely('./vi_dataset/data3.json')
data4 = load_json_list_safely('./vi_dataset/data4.json')
data5 = load_json_list_safely('./vi_dataset/data5.json')
data6 = load_json_list_safely('./vi_dataset/data6.json')
data7 = load_json_list_safely('./vi_dataset/data7.json')
data8 = load_json_list_safely('./vi_dataset/data8.json')
data9 = load_json_list_safely('./vi_dataset/data9.json')
# concat data
current_data = data1
current_data += data2
current_data += data3
current_data += data4
current_data += data5
current_data += data6
current_data += data7
current_data += data8
current_data += data9

# test validate out domain
test_validate_out_side_domain_data = [
    'Trong phản hồi gửi đến báo chí, hãng nhấn mạnh rằng One UI 8 chưa được công bố chính thức và những thiết bị bị phát hiện đang chạy phiên bản này hiện chỉ là các mẫu thử nghiệm nội bộ phục vụ nghiên cứu phát triển, chưa sẵn sàng cho người dùng phổ thông.',
    "Đây là lớp phòng vệ quan trọng bởi nếu trình duyệt Chrome hoạt động với quyền cao nhất, mọi phần mềm độc hại tải xuống thông qua trình duyệt sẽ có thể xâm nhập sâu vào hệ thống mà không gặp cảnh báo.",
    "Lĩnh vực robot đã chứng kiến những bước tiến vượt bậc trong vài năm qua. Từ việc chỉ xuất hiện trong các nhà máy với vai trò cố định, hiếm khi tương tác với con người, robot giờ đây ngày càng trở nên phổ biến và hữu ích hơn trong cuộc sống hàng ngày. Với những nỗ lực nghiên cứu như dự án robot Leva, xu hướng này được dự báo sẽ tiếp tục tăng tốc.",
    "Bên cạnh đó, các phòng thí nghiệm thực hiện kiểm nghiệm hàng xuất khẩu cũng liên tục bị thu hồi. Việc này khiến công tác kiểm nghiệm hàng hóa không được thường xuyên, gây chậm trễ, các doanh nghiệp khó khăn.",
    "Các cầu thủ và người hâm mộ Man United đã quen với việc giành những danh hiệu lớn và đạt được những vị trí cao nhất. Chúng tôi rất biết ơn mọi điều cổ động viên đã nói với cầu thủ và chúng tôi muốn đền đáp người hâm mộ. Các cầu thủ đều biết họ đã ủng hộ chúng tôi nhiều như thế nào và luôn làm những điều tốt nhất cho MU."
]

# shuffle and print test
random.shuffle(current_data)
print(current_data[0][:100])

tokenizer = Tokenizer(
    "./tokenizer",
    smart_split_word_in_noise_word=False
)
tokenizer.load()

print("Kích cở từ vựng:", tokenizer.vocab_size)
print("datat dir:", tokenizer.data_dir)
print("eos token:", tokenizer.eos_token_id)
print("pad token:", tokenizer.pad_token_id)
print("unk token:", tokenizer.unk_token_id)
print("start token:", tokenizer.start_token_id)

# khởi tạo model
"""
MQA là gì?
MQA (Multi Group Attention) là một kỹ thuật giảm heads của Key và Value về 1 head, còn Query vẫn giữ số lượng heads nhất định, nó giúp giảm tài nguyên
và bộ nhớ.

GQA là gì?
GQA (Group Query Attention) gần giống MQA nhưng là một kỹ thuật giảm heads của Key và Vlaue về một số lượng nhỏ hơn Q nhưng không phải bằng 1, nó giúp giảm sử dụng tài nguyên
và bộ nhớ, nhưng vẫn cân bằng được độ chính xác

MHA là gì?
MHA (Multi Heads Attention) là SelfAttention đầy đủ heads, không giảm bớt, chất lượng cao nhất, nhưng tốn tài nguyên.
"""

model_dtype = torch.float32
vocab_size = len(tokenizer.vocab)
model = Causal(
    vocab_size=vocab_size,
    d_model=1024,
    kv_d_model=1024,
    n_q_head=16,
    n_kv_head=16, # giảm KV-HEADS xuống 1 là MQA. Lớn hơn 1 != Q-HEADS là GQA. Bằng Q-HEADS là MHA.
    n_ffn=1024*6,
    n_layer=32,
    device=device,
    dtype=model_dtype
)
model.to(device, model_dtype)

# khởi tạo optimizer
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    eps=1e-5, # epsilon lớn tránh tràn số (bắt buộc khi train fp16)
    weight_decay=0.01 # weight decay kéo trọng số về giá trị nhỏ, tránh overfit
)

# khởi tạo loss, scheduler, scaler
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id,
    label_smoothing=0.1 # label smoothing khiến nhãn đúng bị giảm đi giá trị phân phối, tránh overfit batch
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=6e-5)
scaler = torch.amp.GradScaler()
step_previous_train = 0

def thread_save_model(state_dict, model_path):
    torch.save(state_dict, model_path)

def save_model_multi_thread(model, optim, scheduler, step):
    model_path = r"./pt_models/model.pt"
    optim_path = r"./pt_models/optim.pt"
    scheduler_path = r"./pt_models/scheduler.pt"
    step_path = r"./pt_models/step.pt"
    thread1 = threading.Thread(target=thread_save_model, args=(model.state_dict(), model_path))
    thread2 = threading.Thread(target=thread_save_model, args=(optim.state_dict(), optim_path))
    thread3 = threading.Thread(target=thread_save_model, args=(scheduler.state_dict(), scheduler_path))
    thread4 = threading.Thread(target=thread_save_model, args=(step, step_path))
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    print("Đang đợi lưu model...")
    thread1.join()
    print("Đang đợi lưu optim...")
    thread2.join()
    print("Đang đợi lưu scheduler...")
    thread3.join()
    print("Đang đợi lưu step...")
    thread4.join()
    return

def accumulation_on_step(current_data,
                         tokenizer,
                         steps,
                         batch_size,
                         accumulation_steps,
                         after_epoch_num_to_save_model,
                         test_validate,
                         maxlen,
                         verbose=False):

    global step_previous_train
    # loss
    total_loss = 0.0
    # train on step
    optimizer.zero_grad()
    model_loss_test = "Chưa cập nhật"

    # steps
    for step in range(step_previous_train, steps):

        # khởi tạo data để train
        x = [" ".join(text.split()[:maxlen*3]) for text in current_data[step*batch_size:(step+1)*batch_size]]
        x = tokenizer.batch_encode(x, return_tensor_type="tensor")
        mask = x["attention_mask"].to(device)[:, :maxlen]
        x = x["input_ids"].to(device)[:, :maxlen]

        # khởi tạo mặt nạ cho data train và data train
        model.train()
        xb = x.clone()[:, :-1].to(device)
        yb = x.clone()[:, 1:].to(device)
        mask = mask[:, :-1]

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(xb, mask)
            # mô phỏng batch lớn
            loss = criterion(output.reshape(-1, vocab_size), yb.reshape(-1)) / accumulation_steps

        scaler.scale(loss).backward()

        # accumulation step
        if (step+1) % accumulation_steps == 0:
            step_for_save = step+1
            # optim step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # scheduler step
            scheduler.step()
            # lấy loss model (Loss Test Valiate) có thể là OOD (Outside Domain) bên ngoài các miền data
            model_loss_test = model.get_model_loss(test_input=test_validate, tokenizer=tokenizer)

        # in ra các tham số loss, scheduler lr, shape..vv
        total_loss += loss.item() * accumulation_steps
        print(f'\rStep: {step+1}/{steps}, Loss step: {loss.item() * accumulation_steps}, Loss TestVal: ({model_loss_test}), LR: {scheduler.get_last_lr()}, chiều của dữ liệu train: {x.shape}', end="", flush=True)

        # saving model
        if (step+1) % after_epoch_num_to_save_model == 0:
            print()
            save_model_multi_thread(model, optimizer, scheduler, step_for_save)
            print("Đã lưu model")

    step_previous_train = 0
    return total_loss



"                        ----------------------                "
"                              TRAINING                        "
"                        ----------------------                "

after_epoch_num_to_save_model = 1000
epochs = 1000
batch_size = 2
current_data = current_data
accumulation_steps = 10

print("Lưu ý:")
print(f"Loss TestVal cần {accumulation_steps} ở mỗi step accumulation để update giá trị!")
print(f"Sẽ lưu mô hình sau mỗi {after_epoch_num_to_save_model} steps!")

# backup data
data_train_backup_path="./pt_models/data_train_backup.json"
if not os.path.exists(data_train_backup_path):
    with open(data_train_backup_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

for epoch in range(0, epochs):
    # xáo trộn dữ liệu để training
    random.shuffle(current_data)

    # load backup data
    with open(data_train_backup_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                check_data = line
                break

    if str(check_data) == "[]" or epoch > 0:
        del(check_data)
        with open(data_train_backup_path, "w", encoding="utf-8") as f: json.dump(current_data, f, ensure_ascii=False, indent=4)
        train_data = current_data
    elif str(check_data) != "[]" and epoch == 0:
        del(check_data)
        train_data = load_json_list_safely(data_train_backup_path)

    # train với mô phỏng batch trên step
    steps = (len(train_data) // batch_size)
    total_loss = accumulation_on_step(
        train_data,
        tokenizer,
        steps,
        batch_size,
        accumulation_steps,
        after_epoch_num_to_save_model,
        test_validate_out_side_domain_data,
        maxlen=1024,
        verbose=False
    )
    print()
    print(f"Epoch: {epoch+1}, With loss: {total_loss / steps}")
