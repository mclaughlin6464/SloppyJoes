! -*- f90 -*-
! file lbfgs.f90
! Approximate Hessian inversion
! version 0.1
PROGRAM test_lbfgs
    IMPLICIT NONE
    
    INTEGER row, col, i
    REAL (KIND=8) :: d(3), H_0(3,3),s(3,10), y(3,10)   

    do row=1,3
        do col=1,3
            if(row==col) then
                H_0=1
            else
                H_0=0
            end if
        end do
        d(row) = row 
        do i=1,10
            s(row,i)= 1.0/i 
            y(row,i)= 1.0/i 
        end do
    end do 

    print *,d

    CALL lbfgs(3, H_0, 10, s, y, d)

    print *,d

END PROGRAM test_lbfgs

SUBROUTINE lbfgs(n, H_0, k, s, y, d)

!*****************************************************************
!
!   subroutine lbfgs
!   
!   Approximately apply the inverse Hessian to d.
!   m: dim of H_0 (mxm) and d (m)
!   d: Vector to mupltiply by. On exit, the result of the multiplication
!
!*****************************************************************
    IMPLICIT NONE
    
    INTEGER ::  k, n
    REAL (KIND=8) ::  H_0(n,n), s(n,k), y(n,k), d(n) 

    INTEGER :: i,j
    REAL (KIND=8) ::  a(k), b(k), rho(k) 

    do i=1,k
        !print *, y(:,i), s(:,i)
        !print *,1/DOT_PRODUCT(y(:,i), s(:,i))
        rho(i) = 1.0/DOT_PRODUCT(y(:,i), s(:,i))
    end do

    do i=k, 1,-1
        a(i) =  rho(i)*(DOT_PRODUCT(s(:,i), d))
        d = d - a(i)*y(:,i) ! might need to modify here?
    end do

    d = MATMUL(H_0, d)

    do i=1,k
       b(i) = rho(i)*(DOT_PRODUCT(y(:,i), d)) 
       d = d + (a(i)-b(i))*s(:,i) ! might also need a tweak here
    end do

END SUBROUTINE lbfgs
