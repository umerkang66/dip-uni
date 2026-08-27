function rotate_around_y_centered()
    clc;
    close all;
    % Read image
    I = imread('cup.jpg');
    [rows, cols, ~] = size(I);
    outputView = imref2d([rows cols]);

    % Rotation angle
    theta = deg2rad(30);

    % Half-extents of the image plane, centered at origin
    hw = cols/2;
    hh = rows/2;

    % Camera distance from the plane (controls perspective strength)
    d = max(rows, cols) * 1.5;   % smaller d = stronger perspective, larger d = subtler

    % 4 corners of the image plane in 3D (Z=0, centered at origin)
    corners3D = [-hw  hw  hw -hw;
                 -hh -hh  hh  hh;
                   0   0   0   0];

    % Rotation matrix around Y-axis (vertical axis through the center)
    Ry = [cos(theta)  0  sin(theta);
          0           1  0;
         -sin(theta)  0  cos(theta)];

    rotated3D = Ry * corners3D;

    % Push the plane away from the camera, then perspective-project
    rotated3D(3,:) = rotated3D(3,:) + d;
    f = d;
    cx = cols/2; cy = rows/2;

    proj = zeros(2,4);
    for k = 1:4
        X = rotated3D(1,k); Y = rotated3D(2,k); Z = rotated3D(3,k);
        proj(1,k) = f * X / Z + cx;
        proj(2,k) = f * Y / Z + cy;
    end

    % Original image corners in pixel coordinates
    origCorners = [1 cols cols 1;
                   1 1    rows rows];

    % Fit a projective transform mapping original corners -> rotated corners
    tform_R = fitgeotrans(origCorners', proj', 'projective');

    I_rotated = imwarp(I, tform_R, 'OutputView', outputView);

    % Display
    figure;
    subplot(1,2,1);
    imshow(I);
    title('Original');
    subplot(1,2,2);
    imshow(I_rotated);
    title('Rotated Around Y-axis (centered)');
end
