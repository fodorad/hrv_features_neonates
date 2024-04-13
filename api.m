clear
clc

dataDir = 'data/processed';
infiles = dir(fullfile(dataDir, '*.mat'));
disp(['api.m: Calculating features for ', num2str(length(infiles)), ' files...'])
wb = waitbar(0, '');

for i = 1:length(infiles)
    waitbar(i/length(infiles), wb, sprintf('Progress: %d of %d', i, length(infiles)));
    file = infiles(i).name;
    [~, fileName, ~] = fileparts(infiles(i).name);
    infile = fullfile(dataDir, [fileName '.mat']);
    outfile = fullfile(dataDir, [fileName '_mat_features.csv']);
    data = load(infile);

    if isa(data.rr_interval, 'cell')
    	data.rr_interval = cell2mat(data.rr_interval);
    end
    
    if isa(data.rr_peaks, 'cell')
    	data.rr_peaks = cell2mat(data.rr_peaks);
    end

    if isa(data.rr_interval, 'double') && isvector(data.rr_interval) && size(data.rr_interval, 1) == 1 && ...
        isa(data.rr_peaks, 'double') && isvector(data.rr_peaks) && size(data.rr_peaks, 1) == 1
        hrv_features(data, outfile);
    else
        disp(['[Error]: ', outfile, ' is skipped.'])
    end
end

close(wb);
disp('Done.');