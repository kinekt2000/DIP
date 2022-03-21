import argparse
import numpy as np
import cv2

def largest_rotated_rect_ex(width, height, angle):
    if width <= 0 or height <= 0:
        return 0,0

    width_is_longer = width >= height
    side_long, side_short = (width,height) if width_is_longer else (height,width)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (width*cos_a - height*sin_a)/cos_2a, (height*cos_a - width*sin_a)/cos_2a

    return wr,hr


def largest_rotated_rect(width, height, angle):
    angle = np.radians(angle)
    quadrant = int(np.floor(angle / (np.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else np.pi - angle
    alpha = (sign_alpha % np.pi + np.pi) % np.pi

    bb_w = width * np.cos(alpha) + height * np.sin(alpha)
    bb_h = width * np.sin(alpha) + height * np.cos(alpha)

    gamma = np.arctan2(bb_w, bb_w) if (width < height) else np.arctan2(bb_w, bb_w)

    delta = np.pi - alpha - gamma

    length = height if (width < height) else w

    d = length * np.cos(alpha)
    a = d * np.sin(alpha) / np.sin(delta)

    y = a * np.cos(gamma)
    x = y * np.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def rotate_image(image, angle):
    h, w = image.shape[:2]
    cX, cY = (w/2, h/2)

    matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    
    cos = np.abs(matrix[0,0])
    sin = np.abs(matrix[0,1])

    nw = int(h*sin+w*cos)
    nh = int(h*cos+w*sin)

    matrix[0,2] += (nw/2) - cX
    matrix[1,2] += (nh/2) - cY

    return cv2.warpAffine(image, matrix, (nw, nh))

def crop_image_around_center(image, width, height):
    h, w = image.shape[:2]
    cX, cY = (w//2, h//2)

    x1 = int(cX - width * 0.5)
    x2 = int(cX + width * 0.5)
    y1 = int(cY - height * 0.5)
    y2 = int(cY + height * 0.5)

    return image[y1:y2, x1:x2]

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    return cv2.resize(image, dim, interpolation=inter)



if __name__ == "__main__":
    def coefficient_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} not an integer literal")
        if not 0<x<=1:
            raise argparse.ArgumentTypeError(f"{x} only allowed in range (0..1]")
        return x

    def angle_int(x):
        try:
            x = int(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")
        if not 0<=x<360:
            raise argparse.ArgumentTypeError(f"{x} can't be negative")
        return x

    parser = argparse.ArgumentParser(description="Process bmp or png image")
    parser.add_argument(
        "input_file",
        metavar="input",
        type=str,
        nargs=1,
        help="input image in png or bmp format"
    )
    parser.add_argument(
        "-a", "--angle",
        metavar="integer",
        type=angle_int,
        default=0,
        help="image rotation angle"
    )
    parser.add_argument(
        "-c", "--coef",
        metavar="float(0..1]",
        type=coefficient_float,
        default=1.0,
        help="image stretch ratio"
    )
    parser.add_argument(
        "-s", "--scheme",
        type=int,
        default=1.0,
        choices=range(0,3),
        help="interpolation scheme"
    )

    args = parser.parse_args()

    image = resize_image(cv2.imread(args.input_file[0], flags=cv2.IMREAD_UNCHANGED), width=600)
    h, w = image.shape[:2]

    rotated = rotate_image(image, args.angle)
    cropped = crop_image_around_center(
        rotated,
        *largest_rotated_rect_ex(w, h, args.angle)
    )
    cv2.imshow("original", image)
    cv2.imshow("rotated",  rotated)
    cv2.imshow("cropped",  cropped)
    cv2.waitKey(0)