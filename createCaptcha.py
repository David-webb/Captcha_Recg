#!/usr/bin/python
# -*- coding: utf-8 -*-

# Created by David Teng on 18-7-4

# from captcha.image import ImageCaptcha
#
# img = ImageCaptcha()
# image = img.generate_image('1234')
# image.show()
# image.save('python.jpg')

import Image, ImageDraw, ImageFont, ImageFilter
import random
import time, requests

class mkcaptchas(): 
    # 随机字母:
    def rndChar(self):
        return chr(random.randint(65, 90))
    
    # 随机颜色1:
    def rndColor(self):
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
    
    # 随机颜色2:
    def rndColor2(self):
        return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))


    def runMk(self, filename):
        # 240 x 60:
        width = 60 * 4
        height = 60
        image = Image.new('RGB', (width, height), (255, 255, 255))
        # 创建Font对象:
        font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 36)
        # 创建Draw对象:
        draw = ImageDraw.Draw(image)
        # 填充每个像素:
        for x in range(width):
            for y in range(height):
                draw.point((x, y), fill=self.rndColor())
        # 输出文字:
        char_list = [self.rndChar() for i in range(4)]
        print char_list
        for t in range(4):
            draw.text((60 * t + 10, 10), char_list[t], font=font, fill=self.rndColor2())
        # 模糊:
        # image = image.filter(ImageFilter.BLUR)
        image.save(filename+"".join(char_list)+'.png')

    def getsogoucapture(self, nums=100, basicfname="sogoucapture/"):
        """ 获取搜狗的验证码 """
        for i in range(nums):
            if i % 100 == 0 and i:
                print "获得了%s条....." % i
            timstamp = int(round(time.time() * 1000))
            capturl = "http://weixin.sogou.com/antispider/util/seccode.php?tc=%s" % timstamp
            res = requests.get(capturl)
            with open(basicfname+`i`+'.jpg', 'wb')as wr:
                wr.write(res.content)

        pass

if __name__ == "__main__":

    tpmk = mkcaptchas()
    basicfname = "sogoucaptcha_templates/"

    # 程序生成验证码: 文件名:验证码.jpg
    # for i in range(1):
    #     tpmk.runMk(basicfname)

    # 获取搜狗验证码
    tpmk.getsogoucapture(20000)

    pass