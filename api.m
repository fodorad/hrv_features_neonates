dataDir = 'data/';

infiles = dir(fullfile(dataDir, '*.mat'));

for i = 1:length(infiles)
    file = infiles(i).name;
    [~, fileName, ~] = fileparts(infiles(i).name);
    infile = fullfile(dataDir, [fileName '.mat']);
    outfile = fullfile(dataDir, [fileName '_hrv_features.csv']);
    data = load(infile);
    data.rr_interval = cell2mat(data.rr_interval);
    data.rr_peaks = cell2mat(data.rr_peaks);
    hrv_features(data, outfile)
end