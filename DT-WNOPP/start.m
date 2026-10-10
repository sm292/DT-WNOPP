projectRoot = fullfile(getenv("USERPROFILE"), "Documents", "DT-WNOPP");

folders = {
    "01_MATLAB"
    "02_PYTHON"
    "03_DATA"
    "04_RESULTS"
    "05_PAPER"
};

if ~exist(projectRoot, "dir")
    mkdir(projectRoot);
end

for k = 1:numel(folders)
    folderPath = fullfile(projectRoot, folders{k});
    if ~exist(folderPath, "dir")
        mkdir(folderPath);
    end
end

disp("Project folder created:");
disp(projectRoot);