#!/usr/bin/env python
# coding: utf-8

# **Проект: вариант 1**

# In[54]:


import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import norm, normaltest
from tqdm.auto import tqdm


# **Задание № 1**
# 
# Написать функцию, которая будет считать **retention** игроков (по дням от даты регистрации игрока)
# 

# In[3]:


#Начнем с считывания данных
reg_data = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-e-chernova-25/shared/problem1-reg_data.csv',sep=';')
auth_data = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-e-chernova-25/shared/problem1-auth_data.csv',sep=';')


# *Посмотрим на данные: размер таблицы, типы переменных, наличие пропущенных значений и описательную статистику*

# In[4]:


reg_data


# In[5]:


auth_data


# In[6]:


reg_data.dtypes


# In[7]:


auth_data.dtypes


# In[8]:


reg_data.isna().sum()


# In[9]:


auth_data.isna().sum()


# In[10]:


reg_data.describe()


# In[11]:


auth_data.describe()


# *Также убедимся, что количество уникальных id одинаково в этих таблицах*

# In[12]:


reg_data.reg_ts.nunique()


# In[13]:


auth_data["uid"].nunique()


# *Перейдем к написанию необходимой функции, коротко о ней:*
# 
#     Функция рассчитывает Retention N-ого дня по каждой когорте (дате регистрации пользователя)
#     
#     reg - Датафрейм с данными о регистрации пользователей в игре
#     
#     auth - Датафрейм с данными о входе пользователей в игру
#     
#     from_date - Дата первой когорты
#     
#     to_date - Дата последней когорты
#     
#     retention_days - Дни за которые рассматриваем retention

# In[27]:


def retention(reg, auth, from_date=None, to_date=None):
     
    # Создаем колонки дата регистрации и дата входа в формате datetime

    reg['reg_date'] = pd.to_datetime(reg_data['reg_ts'], unit='s').dt.date
    auth['auth_date'] = pd.to_datetime(auth_data['auth_ts'], unit='s').dt.date
    
    # Отбираем диапазон, который задается в функции
    reg = reg.query('reg_date >= @from_date')
    auth = auth.query('auth_date <= @to_date')

    # Смердживаем две таблицы в один датафрейм
    full_data = pd.merge(auth, reg, on='uid')
    
    # Считаем количество дней с момента регистрации до захода в игру
    full_data['retention_days'] = (full_data.auth_date - full_data.reg_date).dt.days + 1
    
    # Разбиваем на когорты и считаем retention
    cohorts = full_data.groupby(['reg_date', 'retention_days'])['uid'].nunique().reset_index()
    cohorts_1 = cohorts.pivot(index='reg_date', columns='retention_days', values='uid')
    retention = cohorts_1.divide(cohorts_1[1], axis=0).round(3)
    
    # Визуализируем полученные данные
    plt.figure(figsize=(20, 14))
    plt.title('Retention')
    sns.heatmap(data=retention, annot=True, fmt='.0%', vmin=0.0, vmax=0.1, cmap='rocket')
    plt.show()


# In[28]:


retention(reg_data, auth_data, from_date=datetime.date(2020, 9, 1), to_date=datetime.date(2020, 9, 20))


# **Задание № 2**
# 
# 
#     Имеются результаты A/B теста, в котором двум группам пользователей предлагались различные наборы акционных предложений. Известно, что ARPU в тестовой группе выше на 5%, чем в контрольной. При этом в контрольной группе 1928 игроков из 202103 оказались платящими, а в тестовой – 1805 из 202667.
# 
# *Ответить на вопросы:*
#     
#     Какой набор предложений можно считать лучшим? 
#     Какие метрики стоит проанализировать для принятия правильного решения и как?

# In[2]:


#Начнем с считывания данных
df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-e-chernova-25/project_final/Проект_1_Задание_2.csv', sep=';')
df


# In[31]:


control_a = df.query('testgroup == "a"')
test_b = df.query('testgroup == "b"')


# In[17]:


#Отберм 2 датафрейма: контрольный, тестовый
control = test_a.query('revenue > 0')
test = test_b.query('revenue > 0')


# In[18]:


control


# *Посмотрим на такие метрики, как среднее значение, медиану, конверсию, максимальный чек, минимальный чек, ARPU и ARPPU*

# In[11]:


metrics = pd.DataFrame(columns=['test', 'control'])

#Mean    
test_mean = test.revenue.mean()
control_mean = control.revenue.mean()
metrics = metrics.append({'test':test_mean,'control':control_mean}, ignore_index=True)

#Median
test_median = np.median(test['revenue'])
control_median = np.median(control['revenue'])
metrics = metrics.append({'test':test_median,'control':control_median}, ignore_index=True)

#Conversion
test_conv = (test.shape[0]/test_b.shape[0])*100
control_conv = (control.shape[0]/test_a.shape[0])*100
metrics = metrics.append({'test':test_conv,'control':control_conv}, ignore_index=True)

# Максимальный чек
max_check_test = test.revenue.max()
max_check_control = control.revenue.max()
metrics = metrics.append({'test':max_check_test,'control':max_check_control}, ignore_index=True)

# Минимальный чек
min_check_test = test.revenue.min()
min_check_control = control.revenue.min()
metrics = metrics.append({'test':min_check_test,'control':min_check_control}, ignore_index=True)

#ARPU
test_arpu = test_b.revenue.sum()/test_b.shape[0]
control_arpu = test_a.revenue.sum()/test_a.shape[0]
metrics = metrics.append({'test':test_arpu,'control':control_arpu}, ignore_index=True)

#ARPPU
test_arppu = test.revenue.sum()/test.shape[0]
control_arppu = control.revenue.sum()/control.shape[0]
metrics = metrics.append({'test':test_arppu,'control':control_arppu}, ignore_index=True) 

my_index = ['Mean', 'Median', 'Conversion', 'Max', 'Min', 'ARPU', 'ARPPU']
metrics.index = my_index

metrics.apply(lambda x : round(x, 2))


# Теперь посмотрим на распределение чеков в тестовой группе *а* и *b*

# In[32]:


control_a.revenue.hist()
plt.xlabel('Контрольная группа')


# In[33]:


test_b.revenue.hist()
plt.xlabel('Тестовая группа')


# In[34]:


# Проверка распределения на нормальность
stats.shapiro(control_a.revenue.sample(1000, random_state=17))


# In[35]:


# Проверка распределения на нормальность
stats.shapiro(test_b.revenue.sample(1000, random_state=17))


# In[43]:


# Проверка распределения на нормальность (revenue > 0)
stats.shapiro(control.revenue.sample(1000, random_state=17))


# In[44]:


# Проверка распределения на нормальность (revenue > 0)
stats.shapiro(test.revenue.sample(1000, random_state=17))


# *Промежуточный вывод:*
# 
# Две группы не имеют нормального распределения, что видно из визуализаций и p_value < 0.05.
# Так как среди всех игроков и отдельно среди платящих в контрольной группе присутствуют сильно доминирующие моды, мы откажемся от использования U-критерия Манна-Уитни.
# На основе этих выводов для сравнения будем использовать только bootstrap (с np.mean). 
# 

# **ARPU**
# 
# Гипотезы:
# 
#     Н0 - разница в средних значениях в обоих группах отсутствует (при p > 0.05)
# 
#     Н1 - разница есть (при p < 0.05)
# 
# Проверим данные с помощью **bootstrap**
# 

# In[55]:


def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_1), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1)-statistic(samples_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[57]:


bootstrap_test1 = get_bootstrap(control_a.revenue, test_b.revenue)    


# In[58]:


bootstrap_test1['p_value']


# **ARPPU**
# 
# Гипотезы:
# 
#     Н0 - разница в средних значениях в обоих группах отсутствует (при p > 0.05)
# 
#     Н1 - разница есть (при p < 0.05)
#     
# Проверим данные с помощью **bootstrap**

# In[59]:


# Объявим функцию, которая позволит проверять гипотезы с помощью бутстрапа
def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 1000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_1), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1)-statistic(samples_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[61]:


bootstrap_test2 = get_bootstrap(control.revenue, test.revenue)    


# In[62]:


bootstrap_test2['p_value']


# *По результатам тестов мы видим, что p-value > 0.05 и 0 входит в доверительный интервал, что значит, мы не можем отклонить нулевую гипотезу что статистически значимых различий ARPU и ARPPU нет.*

# **Общий вывод по 2 заданию:**
#    
#    По результатам бутстрапа - нулевую гипотезу не отклоняем, так как находится внутри доверительного интервала и p_value > 0.05 
#    
#    Подтверждаем гипотезу о равенстве средних - нет статистически значимых различий.
#    
#    Так, можно сделать вывод, что новое акционное предложение статистически значимо не повлияло на показатели ARPU и ARPPU.
#   
#    
#    Отвечая на вопросы к заданию можно отметить, что особой разницы между тестовой и контрольной групп нет (если смотреть среднюю прибыль). Но существуют различия в объемах платежей (минимальные и максимальные) в контрольной и тестовой группах.
#   
#   *Итог:*
#   
#    По моему мнению, нет смысла на данный момент выкатывать обновление, необходимо проанализировать контрольную группу, понять почему существует такое различие в оплате клиентов и только после этого можно принимать решение.
# 
#     

# **Задание № 3**
# 
# В игре Plants & Gardens каждый месяц проводятся тематические события, ограниченные по времени. В них игроки могут получить уникальные предметы для сада и персонажей, дополнительные монеты или бонусы. Для получения награды требуется пройти ряд уровней за определенное время. С помощью каких метрик можно оценить результаты последнего прошедшего события?
# 
# Предположим, в другом событии мы усложнили механику событий так, что при каждой неудачной попытке выполнения уровня игрок будет откатываться на несколько уровней назад. Изменится ли набор метрик оценки результата? Если да, то как?

# *Метрики, необходимые для оценки результатов последнего пршедшего события:*
# 
#     Дневная аудитория (DAU) — количество уникальных пользователей, которые зашли в приложение в течение суток.
# 
#     Недельная аудитория (WAU) - количество уникальных пользователей, которые зашли в приложение в течение недели.
# 
#     Месячная аудитория (MAU) — количество уникальных пользователей, которые зашли в приложение в течение месяца.
# 
#     Retention - удержание пользователя, которые уже были в игре (процент пользователей, которые возвращались в игру во время тематического события).
#     
#     Revenue - общий доход во время тематического события и вне его, чтобы посмотреть увеличился или нет.
# 
#     Конверсия (CR) — отношение числа пользователей, которые выполнили какое-либо целевое действие к общему числу 
#     пользователей.
#     
#     Customer satisfaction score (CSAT) - опрос после проведенного события об удовлетворенности пользователей.
#     
#     ARPU и ARPPU - если в ивенте есть варианты транзакций для получения 
#     дополнительных наград, бонусов и предметов.
#     
#     Average Number of Sessions - среднее количество сессий (при хороших резултатах должно увеличиться).

# *Метрики, необходимые для оценки результатов после усложения механики игры:*
# 
#     Churn rate - как изменился отток клиентов.
#     
#     Уровень, на котором большинство пользователей переставали играть.
#     
#     Среднее количество пройденных уровней за одну игру.
#     
#     Средняя продолжительность на прохождение уровня.
#     
#     Время, проведенное в игре (при условии, что игроки будут окатываться на предыдущий уровень - продолжительность сессии 
#     будет увеличиваться).
#     
#     Среднее количество откатов.
#     
#     Среднее количество попыток прохождения каждого уровня.
