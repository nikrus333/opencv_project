# opencv_project
https://habr.com/ru/post/467537/ - Библиотеки для декодирования видео. Сравнение на Python и Rust

# Функция фильтрации
'''
kernel = (np.ones)((3, 3), np.float32) / 25 
dst = cv2.filter2D(img, -1, kernel)
'''
# -1 глубина цвета (количество бит для кодирования одного пикселя)
'''
blur = cv2.blur(img, (5,5))
'''
https://intuit.ru/studies/professional_skill_improvements/11289/courses/1105/lecture/17979?page=2
