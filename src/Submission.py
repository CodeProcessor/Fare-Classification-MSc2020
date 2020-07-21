'''
Created on 6/21/20

@author: dulanj
'''
import os


class Submission(object):
    def __init__(self, filename):
        dir_name = 'output'
        self.filename = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.file_pointer = open(self.filename, 'w')
        self.file_pointer.write('tripid,prediction\n')
        print("File created: {}".format(self.filename))

    def write(self, trip_id, pred):
        """:arg
        Write to the submission file
        """
        self.file_pointer.write('{},{}\n'.format(trip_id, 1 if pred == 'correct' else 0))

    def __del__(self):
        """:arg
        Close the file when exit
        """
        print("Saving file: {}".format(self.filename))
        self.file_pointer.close()


if __name__ == "__main__":
    file_name = 'sample_sub.txt'
    obj = Submission(file_name)
    obj.write('123', True)