# sna9
Мои шаги к решению:
1. Запустил прилагаемый бейзлайн, заменив mean на median. Результат улучшился)
2. Далее очищал данные от мусора типа min значение больше max.
3. Потом неоднократно "инжинирил" данные, добавляя такие параметры, как 
- диагональ, отношение высоты к длине и т.п.
- среднеквадратичную ошибку
- iou трушных коробочек
- и т.д.

Результаты оформились в датасеты, лежащие в папке data.

4. Почти сразу решил попробовать найти "центральный бокс" (т.е. с максимальными левыми нижними и минимальными правыми верхними координатами) - но результат удручил. И тут я совершил фатальную ошибку - вариант с МАКСИМАЛЬНЫМИ коробочками даже не стал рассматривать.

5. Потом пытался много разного (классификация, регрессия, кластеризация) - но все впустую (ожидаемо, впрочем).

6. Особенно поразило, что попытка вычислить "хороших" юзеров (т.е. точных) - ничего не дала; видимо, какие-то объекты они обводили тщательно, а какие-то от балды.  
(goodfellas.ipynb)

7. Прорыв случился только тогда, когда я вспомнил про MAX BOXES (maxboxes.ipynb в папке Notebooks. Там же все мои прочие попытки). При этом я совершенно не участвовал ни в каких обсуждениях.

8. Естественно, попытался "подвигать" эти maxboxes влево-вправо, поувеличивать/поуменьшать их и тд. 
НЕ ВЫШЛО.

9. Ну и, собственно решением, принесшим 94-е место - стала попытка вычислить не хороших юзеров, а "хорошие" объекты. Вернее - их характеристики, так, чтобы они не были "выбросами" 
(good 2.ipynb)
- Я вычислил опять всякие диагонали,
- но ТЕПЕРЬ ДЛЯ ТРУ-КООРДИНАТ;
- после чего нашел у них статистику и 
- обкорнал тестовые данные на предмет выбросов из "тру-статистики"

ВСЕ. Больше ничего улучшить не вышло

