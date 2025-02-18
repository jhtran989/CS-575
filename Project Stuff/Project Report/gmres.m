function [ x, e ] = gmres ( A, b, x, max_iterations, threshold )

%*****************************************************************************80
%
%% gmres applies the Generalized Minimum Residual algorithm to solve A*x=b.
%
  n = length ( A );
  m = max_iterations;
%
%  use x as the initial vector
%
  r = b - A * x;

  b_norm = norm ( b );
  error = norm ( r ) / b_norm;
%
%  initialize the 1D vectors
%
  sn = zeros ( m, 1 );
  cs = zeros ( m, 1 );
  e1 = zeros ( n, 1 );
  e1(1) = 1;
  e = [ error ];
  r_norm = norm ( r );
  Q(:,1) = r / r_norm;
  beta = r_norm * e1;
  for k = 1 : m
%
%  Apply the Arnoldi method.
%
    [ H(1:k+1,k), Q(:,k+1) ] = arnoldi ( A, Q, k );
%   
%  Eliminate the last element in H ith row and update the rotation matrix
%
    [ H(1:k+1,k), cs(k), sn(k) ] = apply_givens_rotation ( ...
      H(1:k+1,k), cs, sn, k );
%   
%  Update the residual vector
%
    beta(k+1) = - sn(k) * beta(k);
    beta(k)   = cs(k) * beta(k);
    error     = abs ( beta(k+1) ) / b_norm;
%
%  Save the error
%
    e = [ e; error ];

    if ( error <= threshold )
      break;
    end

  end
%
%  Solve H * y = beta
%
  y = H(1:k,1:k) \ beta(1:k);
%
%  Set x = x + Q * y
%
  x = x + Q(:,1:k) * y;

  return
end
function [ h, q ] = arnoldi ( A, Q, k )

%*****************************************************************************80
%
%% arnoldi applies the Arnoldi algorithm.
%
  q = A * Q(:,k);

  for i = 1 : k
    h(i) = q' * Q(:,i);
    q = q - h(i) * Q(:,i);
  end

  h(k+1) = norm ( q );
  q = q / h(k+1);

  return
end
function [ h, cs_k, sn_k ] = apply_givens_rotation ( h, cs, sn, k )

%*****************************************************************************80
%
%% apply_givens_rotation applies a Givens rotation to H columns.
%
  for i = 1 : k - 1
    temp   =  cs(i) * h(i) + sn(i) * h(i+1);
    h(i+1) = -sn(i) * h(i) + cs(i) * h(i+1);
    h(i)   = temp;
  end
%
%  Update the next sin cos values for rotation
%
  [ cs_k, sn_k ] = givens_rotation ( h(k), h(k+1) );
%
%  Eliminate H(i + 1, i)
%
  h(k) = cs_k * h(k) + sn_k * h(k+1);
  h(k+1) = 0.0;

  return
end
function [ cs, sn ] = givens_rotation ( v1, v2 )

%*****************************************************************************80
%
%% givens_rotation calculates a Givens rotation matrix.
%
  if ( v1 == 0.0 )
    cs = 0.0;
    sn = 1.0;
  else
    t = sqrt ( v1^2 + v2^2 );
    cs = abs ( v1 ) / t;
    sn = cs * v2 / v1;
  end

  return
end

