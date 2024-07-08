dirs = ["seg-gt", "seg-dc"]
ids = ["001", "002"]
lens = [92, 92]

n = length(dirs)
for i=1:1:n
    d = convertStringsToChars(dirs(i))

    m = length(lens)
    for j=1:1:m
        idx = convertStringsToChars(ids(j))
        movieLength = lens(j)

        % Prep segmentations
        Tracker = TracX.Tracker();
        imagePath = fullfile('ctc-data', 'raw', convertStringsToChars(idx));
        segmentationPath = fullfile('ctc-data', d, convertStringsToChars(idx));
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
