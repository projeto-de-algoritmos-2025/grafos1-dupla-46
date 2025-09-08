from PIL import Image, ImageDraw
import cbor2
import sys
import os


def decode_packed_chain_code(packed_integers, total_length):
    if total_length == 0:
        return []
    chain_code = []
    remaining = total_length
    for packed_u32 in packed_integers:
        directions_in_this_u32 = min(16, remaining)
        current = packed_u32
        for _ in range(directions_in_this_u32):
            direction = current & 0x03
            chain_code.append(direction)
            current >>= 2
        remaining -= directions_in_this_u32
        if remaining == 0:
            break
    return chain_code


def scanline_fill(image, points, color):
    if len(points) < 3:
        return

    pixels = image.load()
    width, height = image.size
    min_y = max(0, min(p[1] for p in points))
    max_y = min(height - 1, max(p[1] for p in points))

    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
            if p1[1] <= y < p2[1]:
                if p2[1] - p1[1] != 0:
                    x = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    intersections.append(int(x))

        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = max(0, intersections[i])
                x_end = min(width - 1, intersections[i + 1])
                for x in range(x_start, x_end + 1):
                    pixels[x, y] = color


def decode_cbor_to_image(width, height, contours_data, isolated_pixels_data):
    BACKGROUND_COLOR = 0
    FOREGROUND_COLOR = 255

    image = Image.new("L", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    reconstructed_contours = []

    for contour in contours_data:
        start_point = (contour["start_x"], contour["start_y"])
        points = [start_point]
        current_x, current_y = start_point
        unpacked_directions = decode_packed_chain_code(
            contour["packed_chain_code"], contour["chain_code_length"]
        )
        for direction in unpacked_directions:
            if direction == 0:
                current_x += 1
            elif direction == 1:
                current_y -= 1
            elif direction == 2:
                current_x -= 1
            elif direction == 3:
                current_y += 1
            points.append((current_x, current_y))

        reconstructed_contours.append(
            {"points": points, "is_outer": contour["is_outer"]}
        )

    for contour in reconstructed_contours:
        if contour["is_outer"]:
            scanline_fill(image, contour["points"], FOREGROUND_COLOR)

    for contour in reconstructed_contours:
        if not contour["is_outer"]:
            scanline_fill(image, contour["points"], BACKGROUND_COLOR)

    for contour in reconstructed_contours:
        if len(contour["points"]) > 1:
            draw.line(contour["points"], fill=FOREGROUND_COLOR, width=1)

    pixels = image.load()
    for pixel_data in isolated_pixels_data:
        x, y = pixel_data["x"], pixel_data["y"]
        if 0 <= x < width and 0 <= y < height:
            pixels[x, y] = FOREGROUND_COLOR

    return image


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    cbor_input_file = sys.argv[1]

    try:
        with open(cbor_input_file, "rb") as f:
            data = cbor2.load(f)
    except cbor2.CBORDecodeError:
        sys.exit(1)

    TARGET_WIDTH = data.get("width", 180)
    TARGET_HEIGHT = data.get("height", 135)
    contours_data = data.get("contours", [])
    isolated_pixels_data = data.get("isolated_pixels", [])

    reconstructed_image = decode_cbor_to_image(
        TARGET_WIDTH, TARGET_HEIGHT, contours_data, isolated_pixels_data
    )

    base_name = os.path.splitext(cbor_input_file)[0]
    output_filename = f"{base_name}_decoded.png"
    reconstructed_image.save(output_filename)

    reconstructed_image.show()
