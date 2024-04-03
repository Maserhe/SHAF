import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def _init():  # 初始化
    global image_intermediate_outputs
    global text_intermediate_outputs

    image_intermediate_outputs = torch.empty((0, 1, 768)).to(device, non_blocking=True)
    text_intermediate_outputs = torch.empty((0, 1, 512)).to(device, non_blocking=True)


def set_text_value(value):
    """ 定义一个全局变量 """
    global text_intermediate_outputs  # 声明为全局变量
    if text_intermediate_outputs.shape[0] == 0:
        text_intermediate_outputs = value
    else:
        text_intermediate_outputs = torch.cat((text_intermediate_outputs, value), dim=1)


def set_image_value(value):
    """ 定义一个全局变量 """
    global image_intermediate_outputs  # 声明为全局变量

    if image_intermediate_outputs.shape[0] == 0:
        image_intermediate_outputs = value
    else:
        image_intermediate_outputs = torch.cat((image_intermediate_outputs, value), dim=1)

def get_text_value():
    global text_intermediate_outputs

    # print(text_intermediate_outputs.shape, " === ")
    result = text_intermediate_outputs
    return result


def get_image_value():
    global image_intermediate_outputs
    result = image_intermediate_outputs
    return result


def clear_text():
    global text_intermediate_outputs
    text_intermediate_outputs = torch.empty((0, 1, 512)).to(device, non_blocking=True)


def clear_image():
    global image_intermediate_outputs
    image_intermediate_outputs = torch.empty((0, 1, 768)).to(device, non_blocking=True)


# _init()
#
# x1 = torch.randn((3, 1, 768))
# x2 = torch.randn((3, 1, 768))
# set_image_value(x1)
# set_image_value(x2)
#
# text_shape = get_image_value().shape
# print(text_shape)  # 输出应该是 torch.Size([6, 12, 512])

# x1 = torch.randn((3, 1, 768))
# x2 = torch.randn((3, 1, 768))
#
# print(torch.vstack((x1, x2)).shape)

# x1 = torch.randn((1, 3))
# x2 = torch.randn((1, 1))
# m = torch.nn.Sigmoid()
# s = torch.nn.Softmax(dim=-1)
#
# y = s(x1)
# y2 = s(x2)
#
# print(y[:, 0:1])
# print(y[:, 1:2])
# print(y[:, 2:])

