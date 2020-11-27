from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse import quote_plus
from selenium import webdriver

search=input('검색 : ')
url=f'https://www.google.com/search?q={quote_plus(search)}&rlz=1C1CHZN_koKR926KR926&sxsrf=ALeKk02D62tpvik4raiNBDuxquZbgn-LJQ:1606378974764&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjNgamB5J_tAhXEFogKHfMHBU4Q_AUoAXoECAkQAw&biw=1920&bih=969#imgrc=upw4mNbY0uWHOM&imgdii=xhE_tdSVkbu9iM'

driver=webdriver.Chrome(r"C:\\Users\\chromedriver.exe")
driver.get(url)
for i in range(10000) :
    driver.execute_script("window.scrollBy(0,100000)")

html=driver.page_source
soup=BeautifulSoup(html)
img=soup.select('.rg_i.Q4LuWd')
n=1
imgurl=[]

for i in img :
    try :
        imgurl.append(i.attrs["src"])
    except :
        imgurl.append(i.attrs["data-src"])

for i in imgurl :
    try :
        urlretrieve(i, "./searching/six/"+search+str(n)+".jpg")
        n+=1
        print(imgurl)
    except :
        pass

driver.close()