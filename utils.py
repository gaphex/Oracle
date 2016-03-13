import cPickle
import sys


def dump_obj(obj, fname):
    try:
        f = file(fname, 'wb')
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        return 1
    except Exception as e:
        print e
        return 0

def load_obj(fname):
    try:
        f = file(fname, 'rb')
        loaded_obj = cPickle.load(f)
        f.close()
        return loaded_obj
    except Exception as e:
        print e
        return 0

def progress(i, n, skip = 100, mode = 1):
    if i%skip == 0 and mode == 1:
        sys.stdout.write("\r%s%%" % "{:5.2f}".format(100*i/float(n)))
        sys.stdout.flush()
        if i >= (n/skip - 1)*skip:
            sys.stdout.write("\r100.00%")
    if i%skip == 0 and mode == 2:
        sys.stdout.write("\r%s" % str(i))
        sys.stdout.flush()

