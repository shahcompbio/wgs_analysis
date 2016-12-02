import argparse
import os
import socket
import sys
import subprocess
import time

import design_utils


class BlatServer(object):
    ''' 
    Run a gfServer in a context manager.
    Block until the server is up and accepting connections from clients.
    '''
    def __init__(self, genome_filename):
        self.genome_filename = os.path.abspath(genome_filename)
    def __enter__(self):
        self.gfserver_proc = None
        server_exists = False
        try:
            socket.socket().connect(('localhost', 8899))
            server_exists = True
        except socket.error:
            pass
        if server_exists:
            print 'A gfServer process is already running'
            return
        print 'Running gfServer for genome {}'.format(self.genome_filename)
        print 'To start gfServer directly, run:'
        print 'python {} {}'.format(os.path.realpath(__file__), os.path.realpath(self.genome_filename))
        self.gfserver_proc = subprocess.Popen(['gfServer', 'start', 'localhost', '8899', '-stepSize=5', self.genome_filename + '.2bit'], stdout=subprocess.PIPE)
        with design_utils.TempDirectory() as temps_dir:
            temp_fasta_filename = os.path.join(temps_dir, 'sequence.fa')
            temp_psl_filename = os.path.join(temps_dir, 'align.psl')
            with open(temp_fasta_filename, 'w') as temp_fasta_file:
                temp_fasta_file.write('>1\nCAGGCTTTTAAATTGGCTTTGATGG\n')
            while True:
                try:
                    subprocess.check_call(['gfClient', 'localhost', '8899', '/', temp_fasta_filename, temp_psl_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    break
                except subprocess.CalledProcessError:
                    pass
                sys.stderr.write('Waiting 10s for gfServer to start\n')
                time.sleep(10)
    def __exit__(self ,type, value, traceback):
        if self.gfserver_proc is not None:
            self.gfserver_proc.kill()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('genome', help='genome fasta filename')
    args = argparser.parse_args()
    with BlatServer(args.genome):
        while True:
            time.sleep(100)


