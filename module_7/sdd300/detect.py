from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint_path = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image.
    :param min_score: минимальный порог для детекции.
    :param max_overlap: максимальное перекрытие двух боксов для подавления (NMS).
    :param top_k: если детекций много, оставить только top_k.
    :param suppress: классы, которые нужно подавить.
    :return: аннотированное изображение (PIL Image).
    """

    # Применение трансформаций
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)

    # Forward pass
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Детекция объектов в выходных данных SSD
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores,
        min_score=min_score, max_overlap=max_overlap, top_k=top_k
    )

    # Перенос боксов на CPU и преобразование к оригинальным размерам
    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor([original_image.width, original_image.height,
                                       original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Декодирование меток классов
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # Если объектов не найдено, вернуть исходное изображение
    if det_labels == ['background']:
        return original_image

    # Копируем изображение, чтобы не изменять исходное
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    # Используем шрифт (убедитесь, что файл "calibril.ttf" доступен)
    font = ImageFont.truetype("calibril.ttf", 15)

    # Отрисовка боксов и текста
    for i in range(det_boxes.size(0)):
        if suppress is not None and det_labels[i] in suppress:
            continue

        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        # Рисуем дополнительный прямоугольник для увеличения толщины линии
        offset_box = [l + 1. for l in box_location]
        draw.rectangle(xy=offset_box, outline=label_color_map[det_labels[i]])

        # Получаем размер текста через getbbox (возвращает кортеж (x0, y0, x1, y1))
        bbox = font.getbbox(det_labels[i].upper())
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_location = [box_location[0] + 2., box_location[1] - text_height]
        textbox_location = [box_location[0], box_location[1] - text_height,
                            box_location[0] + text_width + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)

    del draw
    return annotated_image


if __name__ == '__main__':
    img_path = 'BCCD/JPEGImages/BloodImage_00402.jpg'
    original_image = Image.open(img_path, mode='r').convert('RGB')
    annotated_image = detect(original_image, min_score=0.4, max_overlap=0.3, top_k=200)
    annotated_image.show()
