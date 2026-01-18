function [cleanEEG] = removeArtifactsThreshold(eegData, threshold, windowSize)

    nSamples = length(eegData);
    for startIdx = 1:windowSize:nSamples
        endIdx = min(startIdx + windowSize - 1, nSamples);
        window = eegData(startIdx:endIdx);
        % Check if any value exceeds threshold
        if any(abs(window) > threshold)
            artifact_idx = window > threshold;
            window(artifact_idx) = NaN;
            % Truncate the artifact (here set to NaN)
            eegData(startIdx:endIdx) = window;
        end
    end
    % Find samples exceeding threshold in any channel
    cleanEEG = fillmissing(eegData, 'linear');
end