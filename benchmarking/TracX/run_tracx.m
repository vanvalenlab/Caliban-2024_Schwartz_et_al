dirs = ["seg-gt", "seg-dc"]
ids = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012"]
lens = [71, 42, 42, 42, 42, 42, 71, 50, 50, 45, 65, 55]

n = length(dirs)
for i=1:1:n
    d = convertStringsToChars(dirs(i))

    m = length(lens)
    for j=1:1:m
        idx = convertStringsToChars(ids(j))
        movieLength = lens(j)

        % Prep segmentations
        Tracker = TracX.Tracker();
        imagePath = fullfile('data', 'raw', convertStringsToChars(idx));
        segmentationPath = fullfile('data', d, convertStringsToChars(idx));
        segmentationFileNameRegex = 'mask*.tif';
        imageFileNameRegex = 'nuclear*.tif';
        Tracker.prepareDataFromSegmentationMask(imagePath, ...
            segmentationPath, segmentationFileNameRegex, ...
            imageFileNameRegex);
        clear Tracker

        % Configure a tracking project
        projectName = strcat(d, idx);
        fileIdentifierFingerprintImages = 'nuclear';
        fileIdentifierWellPositionFingerprint = [];
        fileIdentifierCellLineage = '';
        imageCropCoordinateArray = [];
        cellDivisionType = 'sym';

        % Create a tracker instance and a new project
        Tracker = TracX.Tracker();
        Tracker.createNewTrackingProject(projectName, imagePath, ...
                segmentationPath, fileIdentifierFingerprintImages, ...
                fileIdentifierWellPositionFingerprint, fileIdentifierCellLineage, ...
                imageCropCoordinateArray, movieLength, cellDivisionType);

        Tracker.runTracker()
        Tracker.runLineageReconstruction('symmetricalDivision', true);

        Tracker.saveTrackerCellCycleResultsAsTable()
        Tracker.saveTrackerResultsAsTable()
    end
end
