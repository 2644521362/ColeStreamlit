import json
import os
from PIL import Image, ImageDraw,ImageFont
import math
import glob  
import time
import skia
import io
import base64
import copy
import torch.nn as nn
import numpy as np
from matplotlib import font_manager
import html
from colorama import Fore, Style, init  
import re
from tqdm import tqdm
import argparse
import torch
import os
import json
from tqdm import tqdm
# import gpt_tool
def is_font_exists(font_name):
    font_list = font_manager.findSystemFonts() 
    for font in font_list:  
        if font_name.lower().replace(' ','-') in font.lower() or font_name.lower().replace(' ','') in font.lower():  
            return True, font  
    return False, False 
intension2json_prompt='''

You are a graphic design content generation robot, and your goal is to output corresponding content in json format based on user input.
The user's input is a paragraph. You need to analyze this paragraph and generate a json. The json format is as follows:
{"canvas_width": 1080, "canvas_height": 1080, "category": "-1", "layers": {"image": {"prompt": "-1"}, "textlayer": {"heading" : [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle": -1, "letter_bold": -1, "letter_spacing": -1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": - 1, "color": [-1, -1, -1]}], "subheading": [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle":-1, "letter_bold": -1, "letter_spacing":-1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": -1, "color": [-1, -1, -1]}], "body": []}}}
In the output json, numbers other than -1 represent the default value, and you don't need to think about it. If the value is -1, it means you need to think about it.
In layers, the prompt of image is the graphic design content, such as:
Illustrate an array of classical musical instruments, like violins, cellos, and flutes, seamlessly integrated with elements of the Amazon wilderness. These instruments should be adorned with delicate representations of Amazonian wildlife, such as tiny tree frogs, colorful parrots, and intricate vines and leaves, as if they are part of the instruments themselves. The background should feature a lush, dense rainforest with a hint of a mystical, foggy ambiance, suggesting the deep, mysterious heart of the Amazon. Include elegant, classic typography for the concert details , blending seamlessly with the natural and musical themes of the poster.
This paragraph will be used as a prompt for the image generation model.
In the textlayer of layers, there are three sub-layers, namely heading, subheading, and body. The content format of these three layers is the same. They are all a list. The text inside is the text content, font is the font type, font_size is the font size, and text_align is the alignment. Method, you can choose from (left, center, right), capitalize is 0 or 1 to determine whether the first letter is capitalized, letter_bold is 0 or 1 to determine whether the font is bold, letter_spacing is 0, and width is the font boundingbox Width, height is the height of the font boundingbox, left is the x-axis coordinate of the upper left corner of the font boundingbox, which is a value between (0,1), which represents the proportion of the distance to the right from the upper left corner of the canvas to the width of the canvas. top is The y-axis coordinate of the upper left corner of the font boundingbox is a value between (0,1), which represents the proportion of the downward distance starting from the upper left corner of the canvas to the height of the canvas. Opacity is transparency, 0 or 1, 1 represents opaque, color It is a list of [r, g, b]. Please think about the content of a graphic design based on user input. You only need to return json. Do not use the content to return images.
Please Return Json Only, No other information.
User input:

'''

dall3Prompt='''
    generate an image for my graphic design background.
    Don't draw texts on the image, ignore relative text information.
    prompt is :
    '''
    
textBboxGeneratePrompt='''
You are a graphic design text boundingbox generation robot. Your goal is to output corresponding content in json format based on the user's text content and pictures.
The user's input is a paragraph. You need to analyze this paragraph and generate a json. The json format is as follows:
{"canvas_width": 1080, "canvas_height": 1080, "category": "-1", "layers": {"image": {"prompt": "-1"}, "textlayer": {"heading" : [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle": -1, "letter_bold": -1, "letter_spacing": -1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": - 1, "color": [-1, -1, -1]}], "subheading": [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle":-1, "letter_bold": -1, "letter_spacing":-1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": -1, "color": [-1, -1, -1]}], "body": []}}}
In the output json, numbers other than -1 represent the default value, and you don't need to think about it. If the value is -1, it means you need to think about it.
In layers, the prompt of the image is the graphic design content, and the corresponding image will be input by the user. You need to think about where to place the text in the image for the most reasonable arrangement.
In the textlayer of layers, there are three sub-layers, namely heading, subheading, and body. The content format of these three layers is the same. They are all a list. The text inside is the text content, font is the font type, font_size is the font size, and text_align is the alignment. Method, you can choose from (left, center, right), capitalize is 0 or 1 to determine whether the first letter is capitalized, letter_bold is 0 or 1 to determine whether the font is bold, letter_spacing is 0, and width is the font boundingbox Width is a value between (0,1), representing the proportion of the distance to the right occupying the width of the canvas. Height is the height of the font boundingbox, and is a value between (0,1), representing the distance downward occupying the height of the canvas. The ratio of The y-axis coordinate is a value between (0,1), which represents the proportion of the downward distance from the upper left corner of the canvas to the height of the canvas. Opacity is transparency, 0 to 255, 255 represents opacity, and color is a [r,g ,b] list, please think about the content of a graphic design based on user input, you only need to return json.
User input:

'''
refinePrompt='''
You are a graphic design text boundingbox refine robot. Your goal is to output corresponding content in json format based on the user's json and pictures.
The user's input is a paragraph. You need to analyze this paragraph and generate a json. The json format is as follows:
{"canvas_width": 1080, "canvas_height": 1080, "category": "-1", "layers": {"image": {"prompt": "-1"}, "textlayer": {"heading" : [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle": -1, "letter_bold": -1, "letter_spacing": -1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": - 1, "color": [-1, -1, -1]}], "subheading": [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle":-1, "letter_bold": -1, "letter_spacing":-1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": -1, "color": [-1, -1, -1]}], "body": []}}}

In layers, the prompt of the image is the graphic design content, and the corresponding image will be input by the user. You need to think about where to place the text in the image for the most reasonable arrangement.
In the textlayer of layers, there are three sub-layers, namely heading, subheading, and body. The content format of these three layers is the same. They are all a list. The text inside is the text content, font is the font type, font_size is the font size, and text_align is the alignment. Method, you can choose from (left, center, right), capitalize is 0 or 1 to determine whether the first letter is capitalized, letter_bold is 0 or 1 to determine whether the font is bold, letter_spacing is 0, and width is the font boundingbox Width is a value between (0,1), representing the proportion of the distance to the right occupying the width of the canvas. Height is the height of the font boundingbox, and is a value between (0,1), representing the distance downward occupying the height of the canvas. The ratio of The y-axis coordinate is a value between (0,1), which represents the proportion of the downward distance from the upper left corner of the canvas to the height of the canvas. Opacity is transparency, 0 to 255, 255 represents opacity, and color is a [r,g ,b] list, please think about the content of a graphic design based on user input, you only need to return json.
The image input by the user is designed according to the json input by the user. Please carefully consider whether the textlayer content in layers needs to be modified. You cannot modify the text content, but can only modify other attributes.
User input:


'''
template_json = {"canvas_width": 1080, "canvas_height": 1080, "category": "-1", "layers": {"image": {"prompt": "-1"}, "textlayer": {"heading": [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle": -1, "letter_bold": -1, "letter_spacing": -1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": -1, "color": [-1, -1, -1]}], "subheading": [{"text": "-1", "font": "-1", "font_size": -1, "text_align": "-1", "capitalize": "-1", "angle":-1, "letter_bold": -1, "letter_spacing":-1, "width": -1, "height": -1, "left": -1, "top": -1, "opacity": -1, "color": [-1, -1, -1]}], "body": []}}}
template_json_blank = {"canvas_width": 1080, "canvas_height": 1080, "category": "-1", "layers": {"image": {"prompt": "-1"}, "textlayer": {"heading": [], "subheading": [], "body": []}}}
def parseJsons(url):
    dat = json.load(open(url))
    dat_template = template_json_blank.copy()
    dat_template['layers']['image']['prompt'] = dat['layers']['image']['prompt']
    dat_template['category'] = dat['category']
    tLayers = dat['layers']['textlayer']
    for d in tLayers:
        for item in tLayers[d]:
            item['font'] = -1
            item['font_size'] = -1
            item['text_align'] = -1
            item['capitalize'] =-1
            item['angle'] = -1
            item['letter_bold'] = -1
            item['letter_spacing'] = -1
            item['width'] = -1
            item['height'] = -1
            item['left'] = -1
            item['top'] = -1
            item['opacity'] = -1
            item['color'] = -1
            dat_template['layers']['textlayer'][d].append(item)
            
    print(dat_template)
# parseJsons()

def RenderText(canvas, textdict):
    
    textdata = textdict
    color = tuple(textdata['color'])
    color = tuple([max(0,min(255,x)) for x in list(color)])
    alpha = 255
    fontname = textdata['font']
    fontsize = textdata['font_size']
    capitalize = textdata['capitalize']
    text_left, text_top, text_width, text_height = textdata['left'], textdata['top'], textdata['width'], textdata['height']
    text_align = textdata['text_align']
    letter_bold = textdata['letter_bold']
    angle = textdata['angle']
    # surface_save = skia.Surface(int(text_width*2),int(text_height*2))
    # canvas_save = surface_save.getCanvas()

    paint = skia.Paint()  
    paint.setAntiAlias(True)  
    paint.setColor(skia.ColorSetARGB(alpha, *color)) 
    flag, font = is_font_exists(fontname)
    if not flag:
        print(f'The {fontname} is not exist, Use Arial rather.')
        fontname = 'Arial'
    else:
        pass
    if letter_bold == 1:
        typeface = skia.Typeface.MakeFromName(fontname, skia.FontStyle.Bold())
    else:
        typeface = skia.Typeface.MakeFromName(fontname, skia.FontStyle.Normal())
    font = skia.Font(typeface, fontsize) 
    text = textdata['text']
    # print(text)
    text = html.unescape(text)
    if capitalize == True or capitalize == 1 or capitalize == 'true' :
        text = text.upper() 
 
    lines = text.strip().split('\n')  
    wrapped_lines = []  
        
    for line in lines:  
        words = line.split(' ') 
            
        current_line = []  
    
        for word in words:  
            test_line = ' '.join(current_line + [word])  
            cur_width = font.measureText(test_line)  
            if cur_width <= text_width*1.05 or (((len(wrapped_lines)+1)*font.getSize() / text_height) > (cur_width/text_width)):  
                current_line.append(word)  
            else:  
                wrapped_lines.append(' '.join(current_line))  
                current_line = [word] 
                 
        wrapped_lines.append(' '.join(current_line))  

        text_y = text_top - font.getMetrics().fAscent 
    with skia.AutoCanvasRestore(canvas):
        
        if angle != 0:
            degree = 180. * angle / np.pi
            canvas.rotate(degree, text_left + text_width / 2., text_top + text_height / 2.)   
        y_offset = 0  
        for line in wrapped_lines: 
            if line == '':
                continue
            print('TEST : ',line)
            blob = skia.TextBlob.MakeFromString(line, font)  
            bounds = blob.bounds()
    
            if text_align == "center":  
                text_x = text_left - bounds.x() + (text_width - bounds.width())/ 2 

            elif text_align == "right":  
                text_x = text_left - bounds.x() + (text_width - bounds.width())  
            else:  
                text_x = text_left - bounds.x()   
            canvas.drawString(line, text_x, text_y+y_offset, font, paint)   
            y_offset += font.getSize()  
    return canvas 

def Render2(image, textlist):
    width, height = image.size
    image_np = np.array(image)
    image_np = np.dstack([image_np, np.ones((image_np.shape[0], image_np.shape[1]), dtype='uint8') * 255]) 
    skia_image = skia.Image.fromarray(image_np)
    surface = skia.Surface(int(width), int(height))
    with surface as canvas:
        canvas.drawImage(skia_image, 0, 0)
        for text in textlist:
            RenderText(canvas, text)
        image = surface.makeImageSnapshot()
        data = image.encodeToData()   
        
        with open('/openseg_blob/PD/lcx/gpt4vCompare/modiy/onlytest.png', 'wb') as file:  
            file.write(data)
        render_img= Image.open(io.BytesIO(base64.b64decode(base64.b64encode(data).decode('UTF-8'))))
    return render_img

def Render(image, textlist, index, save_url,suffix=None):
    width, height = image.size
    image_np = np.array(image)
    image_np = np.dstack([image_np, np.ones((image_np.shape[0], image_np.shape[1]), dtype='uint8') * 255]) 
    skia_image = skia.Image.fromarray(image_np)
    surface = skia.Surface(int(width), int(height))
    with surface as canvas:
        canvas.drawImage(skia_image, 0, 0)
        for text in textlist:
            RenderText(canvas, text)
        renderedimg = surface.makeImageSnapshot()
        data = renderedimg.encodeToData()
        if suffix is not None:
            pngname = f"{index}_{suffix}.png"
        else:
            pngname = f"{index}.png"
        with open(os.path.join(save_url, pngname), 'wb') as file:  
            file.write(data)
    return

'''

'''

def get_image(bg_path, obj_path, example):
    example = copy.deepcopy(example)
    canvas_width = example['canvas_width']
    canvas_height = example['canvas_height']
    flag = example['layers']['objlayer']['flag']
    scale = 1.0
    surface = skia.Surface(int(scale * canvas_width), int(scale * canvas_height))
    # white background color
    bgimg = Image.open(os.path.join(bg_path))
    objimg = Image.open(os.path.join(obj_path))
    objimg = objimg.resize((500,500))
    storeimg = Image.open('/openseg_blob/PD/lcx/gpt4vCompare/target/food.png')
    storeimg = storeimg.resize((500,500))
    with surface as canvas:
        canvas.clear(skia.ColorWHITE)
        
        # bg render
        image = skia.Image.frombytes(
            bgimg.convert('RGBA').tobytes(),
            bgimg.size,
            skia.kRGBA_8888_ColorType)
        rect = skia.Rect.MakeXYWH(0, 0, canvas_width, canvas_height)
        paint = skia.Paint(AntiAlias=True)
        
        with skia.AutoCanvasRestore(canvas):
            canvas.drawImageRect(image, rect, paint=paint)

        # obj render
        # if '1' in flag:
        image  = skia.Image.frombytes(
            storeimg.convert('RGBA').tobytes(),
            storeimg.size,
            skia.kRGBA_8888_ColorType)
        rect = skia.Rect.MakeXYWH(100, 600, 400, 400)
        paint = skia.Paint(AntiAlias=True)
        
        with skia.AutoCanvasRestore(canvas):
            canvas.drawImageRect(image, rect, paint=paint)
            
        image = skia.Image.frombytes(
            objimg.convert('RGBA').tobytes(),
            objimg.size,
            skia.kRGBA_8888_ColorType)
        
        # rect = skia.Rect.MakeXYWH(0, 0, canvas_width,canvas_height) #canvas_height)
        rect = skia.Rect.MakeXYWH(650, 600, 400, 400) #canvas_width,canvas_height)
        paint = skia.Paint(AntiAlias=True)
        
        with skia.AutoCanvasRestore(canvas):
            canvas.drawImageRect(image, rect, paint=paint)
            
        image = surface.makeImageSnapshot()
        data = image.encodeToData()
        render_img= Image.open(io.BytesIO(base64.b64decode(base64.b64encode(data).decode('UTF-8'))))
    return render_img

def easypaste(bg,obj):
    bgimg = Image.open(bg)  
  
    # 打开要粘贴的图片  
    objimg = Image.open(obj)  
    
    # 指定要粘贴的位置  
    paste_mask = objimg.split()[3].point(lambda i: i > 0 and 255)  
  
    bgimg.paste(objimg, (0, 0), mask=paste_mask)  
    return bgimg
def pipeline(dat,bg,obj,name,saveurl):
    
    
    # bg_obj_url = bgurl
        
        # for k in new_map:
        #     if new_map[k]==int(fid):
        #         template_new_id = k

    bgobjimage = get_image(bg,obj,json.load(open(url)))
    bgobjimage = bgobjimage.convert('RGB').resize((dat['canvas_width'],dat['canvas_height']))
    # draw = ImageDraw.Draw(bgobjimage)
    textlist = []
    for d in dat['layers']['textlayer']:
        for item in dat['layers']['textlayer'][d]:
            item['left'] = item['left']#*dat['canvas_width']
            item['top'] = item['top']#*dat['canvas_height']
            item['width'] =item['width']#dat['canvas_width']
            item['height']=item['height'] #dat['canvas_height']
            # item['opacity'] = 255
            textlist.append(item)

    print(textlist)
    img = Render2(bgobjimage, textlist)
    img.save(f'/openseg_blob/PD/lcx/gpt4vCompare/modiy/{name}.png')
   
def refine():
    dat = json.load(open('/openseg_blob/PD/lcx/gpt4vCompare/gpt4Stage4Jsons/Hisaishi.json'))
    bg_obj_url = '/openseg_blob/PD/lcx/gpt4vCompare/dalle3Stage1Pngs/Hisaishi.png'
        
        # for k in new_map:
        #     if new_map[k]==int(fid):
        #         template_new_id = k

    bgobjimage = Image.open(bg_obj_url)
    bgobjimage = bgobjimage.convert('RGB').resize((dat['canvas_width'],dat['canvas_height']))
    draw = ImageDraw.Draw(bgobjimage)
    textlist = []
    for d in dat['layers']['textlayer']:
        for item in dat['layers']['textlayer'][d]:
            item['left'] = item['left']*dat['canvas_width']
            item['top'] = item['top']*dat['canvas_height']
            item['width'] *=dat['canvas_width']
            item['height']*=dat['canvas_height']
            item['opacity'] = 255
            textlist.append(item)
            # for i in range(3): 
            #     x = item['left']
            #     y = item['top']
            #     width = item['width']
            #     height = item['height']
                
                
                # draw.rectangle([x + i, y + i, x + width - i, y + height - i], outline ="red")
    # bgobjimage.save('/openseg_blob/PD/lcx/gpt4vCompare/boxesPngs/test.png')
    Render(bgobjimage, textlist, 1, suffix=None)

def parseJsonForDalle(url):
    dat = json.load(open(url))
    testStr = ""
    idx=1
    
    dall3Prompt='''generate an image for me.\nprompt is :'''
    for d in dat['layers']['textlayer']:
        for item in dat['layers']['textlayer'][d]:
            testStr+=f"{idx} : '{item['text']}'\n"
            idx+=1
    
    print(dall3Prompt+f"\n{dat['layers']['image']['prompt']}\nImage must containing text :\n{testStr} ")
    
    
# parseJsonForDalle('/openseg_blob/PD/lcx/gpt4vCompare/gpt4Stage1Jsons/10.json')
# parseJsons('/openseg_blob/PD/lcx/gpt4vCompare/gpt4Stage1Jsons/10.json')

pipeline_number = 10
pipeline(f'/openseg_blob/PD/lcx/gpt4vCompare/target/64747e260ae834675c59db35.json',f'/openseg_blob/PD/lcx/gpt4vCompare/target/64747e260ae834675c59db35_bg.png','/openseg_blob/PD/lcx/gpt4vCompare/target/grocery.png',
#          pipeline_number,'/openseg_blob/PD/lcx/gpt4vCompare/modiy')

# def realpipeline():
#     idx = 9 
#     intensions = json.load(open('/openseg_blob/PD/lcx/make_eval/temp/intensions_benchmark.json'))
#     intensions = sorted(intensions)
#     for intension in intensions:
#         stage1_prompt = intension2json_prompt+intension

        
#         # dat = gpt_tool.ask_gpt4(stage1_prompt).json()['choices'][0]['message']['content']

#         # with open(f'/openseg_blob/PD/lcx/gpt4vCompare/gpt4Stage1Jsons/{idx}.json','w') as f:
#         #     json.dump(dat,f)
        
#         jsondat = json.load(open('/openseg_blob/PD/lcx/gpt4vCompare/gpt4Stage1Jsons/9.json'))
#         # jsondat = json.loads(dat)
#         #################### DALLE ONLY
#         idx=1
#         testStr=""
#         dall3Prompt='''generate an image for me.\nprompt is :'''
#         print(jsondat)
#         for d in jsondat['layers']['textlayer']:
#             for item in jsondat['layers']['textlayer'][d]:
#                 testStr+=f"{idx} : '{item['text']}'\n"
#                 idx+=1
        
#         dall3Prompt = dall3Prompt+f"\n{jsondat['layers']['image']['prompt']}\nImage must containing text :\n{testStr} "
#         dalle3OnlyImg = gpt_tool.ask_gpt4(dall3Prompt)
#         print(dall3Prompt)
#         print(dalle3OnlyImg)
#         break
# realpipeline()

# ['Create a series of posters for a fictional Ancient Greek Olympics, with each poster highlighting a different sport like chariot racing or discus throw, styled in traditional Greek pottery art.',
#  'Create a customer testimonial advertisement showcasing a pink clothing collection. The advertisement includes a positive review from Jenny Wilson and contact information for the website and phone number: 123-456-789, WWW.FASHIONSTYLE.COM, 123 Anywhere ST., Any City.',
#  "Design a gift certificate for a travel insurance company that offers a 30% discount for all kinds of travel. ",
#  ]

# temp_d = json.load(open('/openseg_blob/PD/lcx/make_eval/template.json'))
# print(temp_d)