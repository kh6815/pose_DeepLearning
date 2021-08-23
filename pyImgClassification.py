from urllib.request import urlopen
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import socket
import sys
import json
import re


def has_css_selector(select):
    try:
        driver.find_element_by_css_selector(select)
        return True
    except:
        return False


def get_content(driver, index):
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')


    # 게시물 업로드 날짜 크롤링
    try:
        date = soup.select('time._1o9PC.Nzb55')[0]['datetime'][:10]
        # print(date[0:4])
        # print(date[5]+date[6])
        # print("게시물 업로드 날짜 : {}".format(date))
    except:
        # 게시물 업로드 날짜 크롤링 실패
        date = 'NULL'
        # print("날짜 크롤링 NULL")

    # 게시물의 사진 크롤링
    try:
        src = []
        fileName = []
        dirPath = []
        # 게시물에 사진이 한장만 있는 경우
        if has_css_selector('div._97aPb > div > div > div.KL4Bh'):
            img = soup.select_one('div.zZYga > div > article > div._97aPb > div > div > div.KL4Bh > img')["src"]
            src.append(img)
            with urlopen(img) as f:
                createFolder('C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/{}/{}/{}'.format(d_insta_id, date[0:4], date[5] + date[6]))
                imgName = d_insta_id + "_" + str(index + 1) + '.jpg'
                dirPathName = "/" + d_insta_id + "/" + date[0:4] + "/" +date[5] + date[6] + "/" + imgName
                with open('C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/{}/{}/{}/'.format(d_insta_id, date[0:4], date[5] + date[6]) + imgName,
                          'wb') as h:
                    imgFile = f.read()
                    h.write(imgFile)
                    fileName.append(imgName)
                    dirPath.append(dirPathName)
                    print("이미지 저장 : {}".format(img))

        # 게시물에 사진이 한장만 있고 인물 태그가 붙어있는 경우
        elif has_css_selector('div._97aPb > div > div > div.eLAPa._23QFA'):
            img = \
            soup.select_one('div.zZYga > div > article > div._97aPb > div > div > div.eLAPa._23QFA > div.KL4Bh > img')[
                "src"]
            src.append(img)
            with urlopen(img) as f:
                createFolder('C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/{}/{}/{}'.format(d_insta_id, date[0:4], date[5] + date[6]))
                imgName = d_insta_id + "_" + str(index + 1) + '.jpg'
                dirPathName = "/" + d_insta_id + "/" + date[0:4] + "/" +date[5] + date[6] + "/" + imgName
                with open('C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/{}/{}/{}/'.format(d_insta_id, date[0:4], date[5] + date[6]) + imgName,
                          'wb') as h:
                    imgFile = f.read()
                    h.write(imgFile)
                    fileName.append(imgName)
                    dirPath.append(dirPathName)
                    print("태그 이미지 저장 : {}".format(img))


        # 게시물에 사진이 여러장 있는 경우
        elif has_css_selector(
                'div._97aPb > div > div.pR7Pc > div.Igw0E.IwRSH.eGOV_._4EzTm.O1flK.D8xaz.fm1AK.TxciK.yiMZG > div > div > div > ul'):
            src, fileName, dirPath = get_image_list(soup, index, date)

        # 게시물에 비디오 영상이 한개 있는 경우
        else:
            print("해당 게시물은 영상(비디오)입니다.")
    except:
        # 게시물의 이미지 크롤링 실패
        print("게시물 이미지 정보 NULL")

    tag_data = []
        # 해쉬태그 크롤링
    try: 
        # 해쉬태그 데이터 기록
        hashdata = driver.find_element_by_css_selector('.C7I1f.X7jCj')
        tag_raw = hashdata.text
        tag = re.findall('#[A-Za-z0-9가-힣]+', tag_raw)
        tag = ''.join(tag).replace("#","#") # "#" 제거
        # '"세나클", "카페"'
        tag_data = tag
    except:
        tag_data = "#태그없음"

    # 크롤링한 정보 반환
    data = [src, date, fileName, dirPath, tag_data]
    return data


def get_image_list(soup, index, date):
    img_list = []
    img_name_list = []
    img_path_list = []
    file = None
    file_type = None
    num = len(soup.select('div.JSZAJ._3eoV-.IjCL9.WXPwG > div.Yi5aA'))
    for idx2 in range(num):
        div = soup.find("div", {"class": "_97aPb"})
        print(div)
        ul = div.find("ul", {"class": "vi798"})
        lis = ul.find_all("li", {"class": "Ckrof"})
        for idx3 in range(len(lis)):
            if lis[idx3].find("img", {"class": "FFVAD"}) is not None and idx3 < len(lis) - 1:
                file_type = "img"
                file, file_type = compare_to_the_following_index(file, file_type, lis, idx3)
                if file_type == "img":
                    break
            elif lis[idx3].find("video", {"class": "tWeCl"}) is not None and idx3 < len(lis) - 1:
                file_type = "video"
                file, file_type = compare_to_the_following_index(file, file_type, lis, idx3)
                if file_type == "img":
                    break
            else:
                print("파일 형식 찾기 불가")
                print(lis[idx3])

        if file is not None and file_type == "img":
            img_list.append(file)
            with urlopen(file) as f:
                createFolder('C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/{}/{}/{}'.format(d_insta_id, date[0:4], date[5] + date[6]))
                imgName = d_insta_id + "_" + str(index + 1) + '-' + str(idx2 + 1) + '.jpg'
                dirPathName = "/" + d_insta_id + "/" + date[0:4] + "/" +date[5] + date[6] + "/" + imgName
                with open(
                        'C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/{}/{}/{}/'.format(d_insta_id, date[0:4], date[5] + date[6]) + imgName, 'wb') as h:
                    imgFile = f.read()
                    h.write(imgFile)
                    img_name_list.append(imgName)
                    img_path_list.append(dirPathName)
                    print("{}번째 이미지 저장 : {}".format(idx2 + 1, file))
        else:
            print("이미지 저장 실패 파일 : {}".format(file))
            print("이미지 저장 실패 파일 타입 : {}".format(file_type))

        if has_css_selector('button._6CZji') and idx2 < num - 1:
            move_next(driver, "next_img")
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            print("{} -> {}".format(idx2 + 1, idx2 + 2))
        else:
            return img_list, img_name_list, img_path_list


def compare_to_the_following_index(file, file_type, lis, idx):
    if file_type == "img":
        file_src = lis[idx].find("img", {"class": "FFVAD"})["src"]
        # print("현재 인덱스의 사진 리스트(이미지) : {}".format(file_src))
    else:
        file_src = lis[idx].find("video", {"class": "tWeCl"})["src"]
        print("현재 인덱스의 사진 리스트(비디오) : {}".format(file_src))

    if file is None:
        file = file_src
        # print("초기 사진 저장")
    else:
        if file == file_src:
            if lis[idx + 1].find("img", {"class": "FFVAD"}) is not None:
                file = lis[idx + 1].find("img", {"class": "FFVAD"})["src"]
                # print("다음 인덱스의 사진 리스트(이미지) : {}".format(file))
                file_type = "img"

            elif lis[idx + 1].find("video", {"class": "tWeCl"}) is not None:
                file = lis[idx + 1].find("video", {"class": "tWeCl"})["src"]
                print("다음 인덱스의 사진 리스트(비디오) : {}".format(file))
                file_type = "video"

    return file, file_type


def insta_searching(id):
    url = 'https://www.instagram.com/' + id + '/'
    return url


def move_next(driver, type):
    if type == "next_img":
        driver.find_element_by_css_selector('button._6CZji').click()
    elif type == "next_post":
        driver.find_element_by_css_selector('a._65Bje.coreSpriteRightPaginationArrow').click()
    time.sleep(3)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def client_socket(d_insta_id, d_insta_pw) :
    # 서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
    HOST = '127.0.0.1'  
    # 서버에서 지정해 놓은 포트 번호입니다. 
    PORT = 8000       

    # 소켓 객체를 생성합니다. 
    # 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.  
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 지정한 HOST와 PORT를 사용하여 서버에 접속합니다. 
    client_socket.connect((HOST, PORT))

    # 메시지를 수신합니다. 
    data = client_socket.recv(1024)
    string_splitted = repr(data.decode()).split()
    insta_id = string_splitted[0]
    insta_pw = string_splitted[1]
    d_insta_id = insta_id.replace("'", "")
    d_insta_pw = insta_pw.replace("'", "")
    print("소켓 ID : {}, PW : {}".format(d_insta_id, d_insta_pw))

    #print('Received from', "d_insta_id-----------------------------")        
    #print('Received from', d_insta_id)
    #print('Received from', d_insta_pw)
    #print('Received', repr(data.decode()))

    # 소켓을 닫습니다.
    client_socket.close()

    return d_insta_id , d_insta_pw


if __name__ == '__main__':
    #d_insta_id ="ykhykh3587@naver.com"
    #d_insta_pw ="rkd!gus9237"
    
    d_insta_id =""
    d_insta_pw =""
    d_insta_id , d_insta_pw = client_socket(d_insta_id, d_insta_pw)

    driver = webdriver.Chrome("C:/Users/pc/Desktop/캡스톤/chromedriver_win32/chromedriver.exe")
    address = 'https://www.instagram.com'
    driver.get(address)
    time.sleep(3)

    login_section = '//*[@id="loginForm"]/div'
    driver.find_element_by_xpath(login_section).click()
    time.sleep(3)
    elem_login = driver.find_element_by_name("username")
    elem_login.clear()

    # 아이디
    elem_login.send_keys(d_insta_id)  # 아이디 입력
    elem_login = driver.find_element_by_name('password')
    elem_login.clear()

    # 비밀번호
    elem_login.send_keys(d_insta_pw)  # 비밀번호 입력력
    time.sleep(3)

    xpath = '//*[@id="loginForm"]/div/div[3]'
    driver.find_element_by_xpath(xpath).click()
    time.sleep(5)

    xpath1 = '//*[@id="react-root"]/section/main/div/div/div/div'
    driver.find_element_by_xpath(xpath1).click()
    time.sleep(5)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    profile = soup.find_all("div", {"class":"Fifk5"})[4]
    print(profile)
    profile_text = profile.find("img")['alt']

    id = profile_text.split('님')[0]
    # id = 'coco20002'
    url = insta_searching(id)
    driver.get(url)
    time.sleep(5)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    span = soup.find("span", {"class": "g47SY"}).text
    # print("span : {}".format(span))

    driver.find_element_by_css_selector('div._9AhH0').click()
    time.sleep(3)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    img_file_name = []
    img_src = []
    img_date = []
    img_url = []
    img_dir_path = []
    img_tag = []

    for idx in range(int(span)):
        data = get_content(driver, idx)
        img_dir_path.append(data[3])
        img_file_name.append(data[2])
        img_src.append(data[0])
        img_date.append(data[1])
        img_url.append(driver.current_url)
        img_tag.append(data[4])
        # print(driver.current_url)
        if idx != int(span) - 1:
            move_next(driver, "next_post")

    data = {
        'img_url': img_url,
        #'img_src': img_src,
        #'img_date': img_date,
        #'img_file_name': img_file_name,
        'img_dir_path' : img_dir_path,
        'img_tag' : img_tag 
    }
    frame = pd.DataFrame(data)
    #frame.to_json("C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/excel/{}.json".format(d_insta_id))
    frame.to_csv("C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/excel/{}.csv".format(d_insta_id), encoding='utf-8-sig')

    '''
    print(client_socket.recv(1024).decode()) #bytes형 데이터를 문자열로 변환 출력
    send_data = json.dumps(data) #dict형 송신 데이터를 JSON으로 직렬화(json string)
    client_socket.send(send_data.encode()) #문자열로된 직렬화 데이터를 bytes로 변환 후 전송
    '''

    time.sleep(3)
    driver.close()

    #sys.exit()
    print("end")

    # print("\n img_src : ")
    # print(img_src)
    # print("\n img date : ")
    # print(img_date)
    # print("\n")
    '''
    January = []
    February = []
    March = []
    April = []
    May = []
    June = []
    July = []
    August = []
    September = []
    October = []
    November = []
    December = []

    for i in range(len(img_url)):
        print("\n{}번째".format(i + 1))
        print(img_url[i], img_src[i], img_date[i])

        month = img_date[i][5] + img_date[i][6]
        if month == "01":
            January.append(img_url[i])
        elif month == "02":
            February.append(img_url[i])
        elif month == "03":
            March.append(img_url[i])
        elif month == "04":
            April.append(img_url[i])
        elif month == "05":
            May.append(img_url[i])
        elif month == "06":
            June.append(img_url[i])
        elif month == "07":
            July.append(img_url[i])
        elif month == "08":
            August.append(img_url[i])
        elif month == "09":
            September.append(img_url[i])
        elif month == "10":
            October.append(img_url[i])
        elif month == "11":
            November.append(img_url[i])
        else:
            December.append(img_url[i])
            '''
