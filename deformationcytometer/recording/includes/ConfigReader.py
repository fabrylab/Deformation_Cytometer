""" pair = value style cfg file reader  for string, int and float values"""

class ConfigReader():
    def __init__(self, file):
        self._file = file
        self._data = dict()


        f = open(file, 'r')

        for l in f.readlines():
            # remove comment lines
            if l.startswith('#'):
                #print("comment line: %s" % l)
                continue

            # remove trailing comment
            if '#' in l:
                l,_,_ = l.rpartition('#') # get only the part befor the #

            # split key and value
            if '=' in l:
                key,sep,value = l.rpartition('=')
                key = key.strip()      # remove white space
                value = value.strip()

                # try convert to correct type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                        value = value

                setattr(self,key,value)
                self._data[key] = value
            else:
                # empty line
                #print("not a key value seperated by = pair: %s" % l)
                continue

    def content(self):
        print("cfg file: %s" % self._file)

        print('key = value')
        print('------------')
        for k,v in self._data.items():
            print(k," = ",v)

