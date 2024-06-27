#coding=utf-8
import argparse
import os
import shutil
import traceback

def move_file(src_path, dst_path, file):
    print 'from : ',src_path
    print 'to : ',dst_path
    try:
        # cmd = 'chmod -R +x ' + src_path
        # os.popen(cmd)
        f_src = os.path.join(src_path, file)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f_dst = os.path.join(dst_path, file)
        shutil.move(f_src, f_dst)
    except Exception as e:
        print 'move_file ERROR: ',e
        traceback.print_exc()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  opt = parser.parse_args()
  parser.add_argument('--workspace', type=str, default='workspace')
  opt.workspace = os.path.join('../results', opt.workspace)
  move_files('./preprocess', opt.workspace )
