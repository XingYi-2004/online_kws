clc;
addpath(".\");
wav_dir = "noisy";
out_dir = "est";
fileList  = dir(wav_dir);
parfor i = 1 : length(fileList)
    if strcmp(fileList(i).name,'.')==1||strcmp(fileList(i).name,'..')==1
        continue;
    end
    wav_path = fullfile(wav_dir, fileList(i).name);
    [filepath, name, ext] = fileparts(fileList(i).name);
    out_path = fullfile(out_dir, name);
    disp(wav_path);
    omlsa(wav_path, char(out_path));
end