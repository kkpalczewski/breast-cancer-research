from ftplib import FTP
import os
import shutil

def copy_if_ics(destination_folder, full_file_path, ftp):
    if full_file_path[-4:] == ".ics":
        destination_file_path = os.path.join(destination_folder, full_file_path.split("/")[-1])
        try:
            ftp.retrbinary("RETR " + full_file_path, open(destination_file_path, "wb").write)
            print("[SUCCESS] Downloaded {}".format(full_file_path))
        except:
            print("[FAILURE] Unable to download {}".format(full_file_path))

def get_ics_data(destination_folder):
    ftp = FTP('figment.csee.usf.edu')
    ftp.login()
    base_path = '/pub/DDSM/cases'
    ftp.cwd(base_path)
    directories1 = ftp.nlst()
    for d1 in directories1:
        d1_path = os.path.join(base_path,d1)
        ftp.cwd(d1_path)
        directories2 = ftp.nlst()
        for d2 in directories2:
            d2_path = os.path.join(d1_path, d2)
            ftp.cwd(d2_path)
            directories3 = ftp.nlst()
            for d3 in directories3:
                d3_path = os.path.join(d2_path, d3)
                ftp.cwd(d3_path)
                files = ftp.nlst()
                for f in files:
                    source_file_path = os.path.join(d3_path, f)
                    copy_if_ics(destination_folder, source_file_path, ftp)

if __name__=="__main__":
    destination_folder = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/ics_files"
    get_ics_data(destination_folder)