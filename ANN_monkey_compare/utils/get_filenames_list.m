function [targetFiles] = get_filenames_list(dir_path, pattern)

allFiles = dir(dir_path);

targetFiles = {};

for i = 1:numel(allFiles)

    if strcmp(allFiles(i).name, '.') || strcmp(allFiles(i).name, '..')
        continue;
    end


    if ~allFiles(i).isdir

        if ~isempty(strfind(allFiles(i).name, pattern))
            targetFiles{end+1} = allFiles(i).name; 
        end
    end
end

targetFiles = natsort(targetFiles)';
disp('filelist:');
for i = 1:numel(targetFiles)
    fprintf('%s\n', targetFiles{i});
end

end