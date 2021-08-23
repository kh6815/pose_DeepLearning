from urllib.request import urlopen
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import os


# 사용자가 크롤링하려는 해시태그 이름으로 URL을 지정하고 해시태그 이름에 '#'를 추가해 줍니다.
def insta_searching(tag):
    url = 'https://www.instagram.com/explore/tags/' + tag
    tag = '#' + tag
    return tag, url


# 해시태그로 검색된 URL로 이동하면 여러 게시물들이 나오는데 첫번째 게시물로 이동합니다.
def select_first(driver):
    first = driver.find_element_by_css_selector("div._9AhH0")
    first.click()
    time.sleep(3)


def get_content(driver, idx1):
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    # 게시물의 태그 크롤링
    try:
        # 게시물 + 댓글 작성자 이름 리스트
        name_list = []
        # 게시물 + 댓글 내 태그 리스트
        tag_list = []
        # 게시물 + 댓굴 접근 div 리스트
        divs = soup.find_all("div", {'class': 'C4VMK'})
        # print("게시물 + 댓글 갯수 : {}개".format(len(divs)))
        if len(divs) > 0:
            # 게시물 + 댓글 내 작성자 이름 저장하기
            for idx in range(len(divs)):
                # 첫번쨰 인덱스는 게시물
                if idx == 0:
                    name_list.append(divs[idx].select_one('h2 > div > span > a').text)
                    print("게시물 작성자 아이디 : {}".format(name_list[0]))
                # 나머지 인덱스는 모두 댓글
                else:
                    name_list.append(divs[idx].select_one('h3 > div > span > a').text)
                    # print("{}번째 댓글 작성자 아이디 : {}".format(idx, name_list[idx]))
            # print("이름 갯수 : {}, 이름 리스트 : {}".format(len(name_list), name_list))
        else:
            print("게시물 + 댓글 리스트 접근 실패!")
            return None
        # 게시물 + 댓글 내 태그 저장하기
        for idx1 in range(len(name_list)):
            # 작성자 이름 리스트를 토대로 게시물 작성자 이름과 일치하는 아이디를 찾는다
            if name_list[0] == name_list[idx1]:
                tags = divs[idx1].find_all("a", {"class": "xil3i"})
                # 게시물 + 댓글 내 태그가 있는경우
                if len(tags) > 0:
                    for idx2 in range(len(tags)):
                        tag_list.append(tags[idx2].text)
                    # 검색 태그가 존재하는경우 반복문 나감
                    if "#" + word in tag_list:
                        break
                # 게시물 + 댓글 내 태그가 없는경우
                else:
                    # 게시물 작성자가 자신의 댓글 내 답글에 태그를 작성할 경우 답글을 찾아보기 위해 답글 보기 버튼을 눌러야한다.
                    if idx1 > 0 and has_css_selector("ul > ul:nth-child("+str(idx1+1)+") > li > ul > li > div > button"):
                        # 답글 보기 버튼 접근을 하였고 버튼 내 텍스트에 접근하여 '답글 보기'와 '답글 숨기기'를 구별함
                        reply_btn = soup.find("span", {"class": "EizgU"}).text
                        # '답글 보기' 되어있을 경우 해당 버튼을 눌러줍니다.
                        if reply_btn != "답글 숨기기":
                            # 해당 댓글의 '답글 보기' 버튼을 눌러줍니다.
                            driver.find_element_by_css_selector("ul > ul:nth-child(" + str(idx1 + 1) + ") > li > ul > li > div > button.sqdOP.yWX7d.y3zKF").click()
                            time.sleep(2)
                            html = driver.page_source
                            soup = BeautifulSoup(html, 'html.parser')
                            # 웹페이지를 새로고침했기때문에 상위 경로부터 다시 지정해줌
                            new_divs = soup.find_all("div", {"class": "C4VMK"})
                            num_of_reply = len(new_divs) - len(divs)
                            # print("답글의 갯수 : {}개".format(num_of_reply))
                            # 답글에서 답글 작성자와 태그를 찾습니다.
                            for idx3 in range(1, num_of_reply + 1):
                                reply_name = new_divs[idx1 + idx3].select_one('h3 > div > span > a').text
                                if name_list[0] == reply_name:
                                    reply_tags = new_divs[idx1 + idx3].find_all("a", {"class": "xil3i"})
                                    if len(reply_tags) > 0:
                                        for idx4 in range(len(reply_tags)):
                                            tag_list.append(reply_tags[idx4].text)
                                        if "#" + word in tag_list:
                                            break
        if len(tag_list) == 0:
            print("게시물 테그를 불러오지 못했습니다.")
            return None
        else:
            print("태그 갯수 : {}개, 태그 : {}".format(len(tag_list), tag_list))

    except:
        # 게시물의 태그 크롤링 실패
        print("게시물 테그 크롤링 NULL")
        return None

    # 게시물의 사진 크롤링
    try:
        # 게시물에 사진이 한장만 있는 경우
        if has_css_selector('div._97aPb > div > div > div.KL4Bh'):
            imgPath = 'div.zZYga > div > article > div._97aPb > div > div > div.KL4Bh > img'
            get_image(soup, imgPath, idx1)

        # 게시물에 사진이 한장만 있고 인물 태그가 붙어있는 경우
        elif has_css_selector('div._97aPb > div > div > div.eLAPa._23QFA'):
            imgPath = 'div.zZYga > div > article > div._97aPb > div > div > div.eLAPa._23QFA > div.KL4Bh > img'
            get_image(soup, imgPath, idx1)

        # 게시물에 사진이 여러장 있는 경우
        elif has_css_selector('div._97aPb > div > div.pR7Pc > div.Igw0E.IwRSH.eGOV_._4EzTm.O1flK.D8xaz.fm1AK.TxciK.yiMZG > div > div > div > ul'):
            get_image_list(soup, idx1)

        # 게시물에 비디오 영상이 한개 있는 경우
        else:
            print("해당 게시물은 영상(비디오)입니다.")
            return None
    except:
        # 게시물의 이미지 크롤링 실패
        print("게시물 이미지 정보 NULL")
        return None

    # 게시물 업로드 날짜 크롤링
    try:
        date = soup.select('time._1o9PC.Nzb55')[0]['datetime'][:10]
        # print("게시물 업로드 날짜 : {}".format(date))
    except:
        # 게시물 업로드 날짜 크롤링 실패
        date = 'NULL'
        print("날짜 크롤링 NULL")

    # 게시물의 좋아요 수 크롤링
    try:
        # 좋아요 수
        if has_css_selector('div.Nm9Fw > a'):
            if has_css_selector('div.Nm9Fw > a > span'):
                like = soup.select_one('div.Nm9Fw > a > span').text
                like = int(like.replace(',', ''))
                # print('좋아요 : {}개'.format(like))
            else:
                like = 1
                # print('좋아요 : {}개'.format(like))
        # 좋아요 수 정보가 없는 경우
        elif has_css_selector('div.Nm9Fw > button'):
            like = 0
            # print('좋아요 정보가 없습니다.')
        # 해당 게시물이 비디오일 경우 조회수
        elif has_css_selector('div.HbPOm._9Ytll > span'):
            like = 'NULL'
            print('좋아요가 아닌 조회수 정보가 있습니다.')
    except:
        # 게시물의 좋아요 수 크롤링 실패
        like = 'NULL'
        print("좋아요 크롤링 NULL")

    # 사용자의 장소 크롤링
    try:
        # 게시물에 장소 정보가 있는 경우
        if has_css_selector('div.JF9hh > a'):
            place = soup.select_one('div.JF9hh > a').text
            # print('장소 : {}'.format(place))
        # 게시물에 장소 정보가 없는 경우
        else:
            place = ''
            # print('장소 정보가 비워져있습니다')
    except:
        # 사용자의 장소 크롤링 실패
        place = 'NULL'
        print("장소 크롤링 NULL")

    # 크롤링한 정보 반환
    data = [date, like, place, tag_list]
    return data


# css에 해당 selector 존재하는지 확인하는 함수입니다.
def has_css_selector(select):
    try:
        driver.find_element_by_css_selector(select)
        return True
    except:
        return False


# 게시물 내 사진이 한장만 있을 경우 해당 사진을 저장하는 함수입니다.
def get_image(soup, imgPath, idx):
    img = soup.select_one(imgPath)["src"]
    with urlopen(img) as f:
        createFolder('./{}'.format(word))
        with open('./{}/'.format(word) + word + str(idx + 1) + '.jpg', 'wb') as h:
            imgFile = f.read()
            h.write(imgFile)
            print("이미지 저장 : {}".format(img))


# 게시물 내 사진이 여러장 있을 경우 해당 사진들을 저장하는 함수입니다.
def get_image_list(soup, idx1):
    file = None
    file_type = None
    # 게시물 내 사진 및 비디오의 갯수를 저장합니다.
    num = len(soup.select('div.JSZAJ._3eoV-.IjCL9.WXPwG > div.Yi5aA'))
    for idx2 in range(num):
        ul = soup.find("ul", {"class": "vi798"})
        lis = ul.find_all("li", {"class": "Ckrof"})
        # 인스타그램 게시물은 사진을 넘길때마다 사진의 인덱스 순서가 변경되며
        # 삽입, 삭제 과정을 지닌 연결리스트 구조를 띄고 있기 때문에 사진 리스트를 매번 불러와야합니다.
        for idx3 in range(len(lis)):
            # 해당 인덱스에 사진이 있는 경우입니다.
            if lis[idx3].find("img", {"class": "FFVAD"}) is not None and idx3 < len(lis)-1:
                file_type = "img"
                file, file_type = compare_to_the_following_index(file, file_type, lis, idx3)
                if file_type == "img":
                    break
            # 해당 인덱스에 비디오가 있는 경우입니다.
            elif lis[idx3].find("video", {"class": "tWeCl"}) is not None and idx3 < len(lis)-1:
                file_type = "video"
                file, file_type = compare_to_the_following_index(file, file_type, lis, idx3)
                if file_type == "img":
                    break
        # 현재 인덱스에 사진 정보가 있는 경우 해당 사진을 저장합니다.
        if file is not None and file_type == "img":
            with urlopen(file) as f:
                createFolder('./{}'.format(word))
                with open('./{}/'.format(word) + word + str(idx1 + 1) + '-' + str(idx2 + 1) + '.jpg', 'wb') as h:
                    imgFile = f.read()
                    h.write(imgFile)
                    print("{}번째 이미지 저장 : {}".format(idx2+1, file))
        # 다음 사진으로 넘어가는 버튼이 있을 경우 다음 사진으로 이동합니다.
        if has_css_selector('button._6CZji') and idx2 < num-1:
            move_next(driver, "next_img")
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            print("{} -> {}".format(idx2+1, idx2+2))


# 현재 인덱스의 파일과 다음 인덱스의 파일을 비교하는 함수입니다.
def compare_to_the_following_index(file, file_type, lis, idx):
    # 현재 인덱스가 사진인 경우 사진 정보를 저장하고 비디오인 경우 비디오 정보를 저장합니다
    if file_type == "img":
        file_src = lis[idx].find("img", {"class": "FFVAD"})["src"]
        # print("현재 인덱스의 사진 리스트(이미지) : {}".format(file_src))
    else:
        file_src = lis[idx].find("video", {"class": "tWeCl"})["src"]
        # print("현재 인덱스의 사진 리스트(비디오) : {}".format(file_src))
    # 파일의 정보가 없는 초기 파일 정보를 저장합니다.
    if file is None:
        file = file_src
        # print("초기 사진 저장")
    else:
        if file == file_src:
            # 다음 인덱스가 사진인 경우 사진 정보를 저장하고 비디오인 경우 비디오 정보를 저장합니다
            if lis[idx + 1].find("img", {"class": "FFVAD"}) is not None:
                file = lis[idx + 1].find("img", {"class": "FFVAD"})["src"]
                # print("다음 인덱스의 사진 리스트(이미지) : {}".format(file))
                file_type = "img"

            elif lis[idx + 1].find("video", {"class": "tWeCl"}) is not None:
                file = lis[idx + 1].find("video", {"class": "tWeCl"})["src"]
                # print("다음 인덱스의 사진 리스트(비디오) : {}".format(file))
                file_type = "video"
    return file, file_type


# 주어진 형식에 따라 페이지를 이동하는 함수입니다.
def move_next(driver, type):
    # type이 "next_img"인 경우 다음 사진으로 이동합니다.
    if type == "next_img":
        driver.find_element_by_css_selector('button._6CZji').click()
    # type이 "next_post"인 경우 다음 게시물로 이동합니다.
    elif type == "next_post":
        driver.find_element_by_css_selector('a._65Bje.coreSpriteRightPaginationArrow').click()
    time.sleep(3)


# 폴더를 추가하는 함수입니다.
def createFolder(directory):
    try:
        # 기존의 디렉토리에 폴더가 존재하지 않은 경우 폴더를 생성해 줍니다.
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


if __name__ == '__main__':
    # Chrome Driver 경로를 지정하여 Chrome Driver 프로그램을 사용합니다.
    driver = webdriver.Chrome("chromedriver.exe")
    # 인스타그램 사이트로 크롬 연결합니다.
    address = 'https://www.instagram.com'
    driver.get(address)
    time.sleep(3)
    # 인스타그램 로그인 페이지 섹션을 연결합니다.
    login_section = '//*[@id="loginForm"]/div'
    driver.find_element_by_xpath(login_section).click()
    time.sleep(3)
    # 주어진 아이디를 로그인 아이디 키값으로 보내줍니다.
    elem_login = driver.find_element_by_name("username")
    elem_login.clear()
    elem_login.send_keys('')# 사용자의 인스타그램 아이디를 추가해 주어야합니다.
    # 주어진 비밀번호를 로그인 비밀번호 키값으로 보내줍니다.
    elem_login = driver.find_element_by_name('password')
    elem_login.clear()
    elem_login.send_keys('')# 사용자의 인스타그램 비밀번호를 추가해 주어야합니다.
    time.sleep(3)
    # 로그인 버튼을 눌러줍니다.
    xpath = '//*[@id="loginForm"]/div/div[3]'
    driver.find_element_by_xpath(xpath).click()
    time.sleep(3)
    # 사용자 계정 정보를 저장하는 알림창을 무시해줍니다.
    xpath1 = '//*[@id="react-root"]/section/main/div/div/div/div'
    driver.find_element_by_xpath(xpath1).click()
    time.sleep(3)
    # insta_searching 함수로 파생된 URL로 이동합니다.
    word = '' # 크롤링할 해시태그 이름을 word변수에 지정해 주어야합니다.
    word, url = insta_searching(word)
    driver.get(url)
    time.sleep(3)
    # select_first 함수로 해시태그 검색된 게시물 리스트 중 첫번째 게시물로 이동합니다.
    select_first(driver)
    # dates 리스트는 게시물의 날짜를 저장하는 리스트 입니다.
    dates = []
    # likes 리스트는 게시물의 좋아요를 저장하는 리스트 입니다.
    likes = []
    # places 리스트는 게시물의 장소를 저장하는 리스트 입니다.
    places = []
    # tags 리스트는 게시물의 태그들을 저장하는 리스트 입니다.
    tags = []
    # target 변수는 크롤링 횟수를 지정하는 변수입니다.
    target = 50
    # 크롤링 횟수만큼 반복하여 크롤링합니다.
    for idx1 in range(target):
        # idx2 변수는 크롤링 실패할 경우 크롤링 횟수에 포함시키지 않기 위해 크롤링 실패 횟수 변수입니다.
        idx2 = 0
        while True:
            # get_content 함수는 게시물 내 정보를 크롤링하는 함수입니다.
            data = get_content(driver, idx1)
            # 크롤링이 성공한 경우입니다.
            if data is not None:
                print('{}-{}번째 크롤링 성공!\n'.format(idx1 + 1, idx2 + 1))
                # 크롤링한 게시물 내 정보를 dates, likes, places, tags 리스트에 추가해줍니다.
                dates.append(data[0])
                likes.append(data[1])
                places.append(data[2])
                tags.append(data[3])
                break
            # 크롤링이 실패한 경우입니다.
            else:
                print('{}-{}번째 크롤링 실패!\n'.format(idx1 + 1, idx2 + 1))
                # move_next 함수는 다음 게시물로 이동합니다.
                move_next(driver, "next_post")
            idx2 = idx2 + 1
        # move_next 함수는 다음 게시물로 이동합니다.
        move_next(driver, "next_post")
    # 크롤링 횟수의 크기만큼 저장된 리스트들을 딕셔너리형식으로 저장합니다.
    data = {
        'date': dates,
        'like': likes,
        'place': places,
        'tags': tags,
    }
    # 딕셔너리형 변수를 csv파일로 변형하여 저장합니다.
    frame = pd.DataFrame(data)
    frame.to_csv("{}.csv".format(word), encoding='utf-8-sig')
