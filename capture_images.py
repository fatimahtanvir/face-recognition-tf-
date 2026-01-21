# capture_images.py
import os
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Label / person name (no spaces)')
    parser.add_argument('--out', default='dataset', help='Output dataset directory')
    parser.add_argument('--num', type=int, default=40, help='Number of images to capture')
    args = parser.parse_args()

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print('Press SPACE to capture, ESC to exit early.')

    count = 0
    while count < args.num:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        cv2.putText(disp, f'{args.name}: {count}/{args.num}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Capture (press SPACE)', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            filename = os.path.join(out_dir, f'{args.name}_{count:03d}.jpg')
            cv2.imwrite(filename, frame)
            print('Saved', filename)
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print('Done.')

if __name__ == '__main__':
    main()

