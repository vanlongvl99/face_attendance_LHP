# facial-recognition-python-django
Face detection and facial recognition along with recognized persons information fetched from database.

General Languages and versions

    •	Django version: 1.11.9
    •	OpenCV version: 3.4.0
    •	Sklearn version: 0.19.1


# Collect data:

Dữ liệu được lưu trong thư mục `dataset1`: ta có thể thay đổi tên thư mục tại file `./facial_recog/faceRecog/views.py`.
Trong thực mục `dataset1` gồm 2 phần là:
- `raw` chứa dữ liệu nguyên gốc (ảnh và video)
- `only_face` chỉ chứa khuôn mặt
![](./image/inside_dataset.png)



Để thu thập dữ liệu ta run 

`python3 video_images_faces.py`

Giao diện hiện lên:
![](./image/collect_data_interface.png)

Thao tác lấy dữ liệu 1 người mới: ta nhập tên, sđt sau đó click và button `Take image`. Nhấn phím `Q` để dừng việc lấy dư liệu.

Click `Quit` để thoát khỏi chương trình thu thập dữ liệu.

# Chương trình điểm danh

- Source code Nhận diện nằm trong file  `/facial_recog/faceRecog/views.py` ở hàm `detect`:

- Vì file model nặng nên không thể upload lên github, các bạn tải file mode [tại đây](https://drive.google.com/drive/folders/1sxYaEGKeChUC4NlozsdAQZT09vQYId8f?usp=sharing). Sau khi download các file model về, ta lưu các file model trong thư mục `./facial_recog/ml/model_file`. Lưu ý không chỉnh sửa tên các file model, nếu không sẽ bị 1 số lỗi gọi tên file khi chạy chương trình.


Run 

`python3 manage.py runserver`


