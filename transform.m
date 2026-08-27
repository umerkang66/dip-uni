function transform()
    clc;
    close all;

    I = imread('cup.jpg');

    [rows, cols, ~] = size(I);
    outputView = imref2d([rows cols]);

    % Scaling matrix
    sx = 1.5;
    sy = 1.5;

    S = [sx  0   0;
         0   sy  0;
         0   0   1];

    % Translation matrix
    tx = 100;
    ty = 50;

    T = [1  0  tx;
         0  1  ty;
         0  0  1];

    % Rotation matrix
    theta = 30;
    theta = deg2rad(theta);

    R = [cos(theta) -sin(theta)  0;
         sin(theta)  cos(theta)  0;
         0           0           1];

    % Scaling
    tform_S = affine2d(S');
    I_scaled = imwarp(I, tform_S, 'OutputView', outputView);

    % Translation
    tform_T = affine2d(T');
    I_translated = imwarp(I, tform_T, 'OutputView', outputView);

    % Rotation
    tform_R = affine2d(R');
    I_rotated = imwarp(I, tform_R, 'OutputView', outputView);

    % Combined transformation
    H = R * T * S;

    tform_H = affine2d(H');
    I_combined = imwarp(I, tform_H, 'OutputView', outputView);

    % Display
    figure;

    subplot(1,5,1);
    imshow(I);
    title('Original');

    subplot(1,5,2);
    imshow(I_scaled);
    title('Scaled');

    subplot(1,5,3);
    imshow(I_translated);
    title('Translated');

    subplot(1,5,4);
    imshow(I_rotated);
    title('Rotated');

    subplot(1,5,5);
    imshow(I_combined);
    title('Combined');
end
