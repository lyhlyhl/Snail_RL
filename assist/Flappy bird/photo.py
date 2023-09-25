from PIL import Image

# 打开图像文件
image = Image.open('bird.png')

# 定义目标大小
new_size = (48, 38)  # 指定新的宽度和高度

# 调整图像大小
resized_image = image.resize(new_size)  # 使用抗锯齿滤镜进行调整

# 保存调整大小后的图像
resized_image.save('resizedbird.png')

# 关闭原始图像
image.close()