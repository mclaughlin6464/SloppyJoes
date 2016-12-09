! -*- f90 -*-
! file lbfgs.f90
! Approximate Hessian inversion
! version 0.1
SUBROUTINE lbfgs(n, H_0, k, s, y, d)

!*****************************************************************
!
!   subroutine lbfgs
!   
!   Approximately apply the inverse Hessian to d.
!   n: dim of H_0 (nxn) and d (n)
!   d: Vector to mupltiply by. On exit, the result of the multiplication
!
!*****************************************************************
    IMPLICIT NONE
    
    INTEGER ::  k, n
    REAL (KIND=8) ::  H_0(n,n), s(n,k), y(n,k), d(n) 

    INTEGER :: i,j
    REAL (KIND=8) ::  a(k), b(k), rho(k) 

!    print *, n,k !    print *,'s',s !    print *,'y',y

    do i=1,k-1
        rho(i) = 1.0/DOT_PRODUCT(y(:,i), s(:,i))
    end do

    do i=k-1, 1,-1
        a(i) =  rho(i)*(DOT_PRODUCT(s(:,i), d))
        d = d - a(i)*y(:,i) ! might need to modify here?
    end do

    print *,'d old',d
    d = MATMUL(H_0, d)
    print *,'d old',d

    do i=1,k-1
       b(i) = rho(i)*(DOT_PRODUCT(y(:,i), d)) 
       !print *,i, b(i), rho(i)
       d = d + (a(i)-b(i))*s(:,i) ! might also need a tweak here
       !print *,'d',d
    end do

END SUBROUTINE lbfgs

SUBROUTINE update_storage(n, k, istep, s, y, d_x, d_jac, info)

    IMPLICIT NONE

    INTEGER :: n,k,istep

    REAL(KIND=8) :: s(n,k), y(n,k)
    REAL(KIND=8) :: d_x(n), d_jac(n)

    INTEGER :: i, info 

    info = 1
    do i=1,n
        print *,'d_x', d_x(i)
        print *,'d_x', d_x(i) .LE. 1.0E-015 
        if (ABS(d_x(i)) .LE. 1.0E-015) then
            info = 0
            print *,'cont', info 
        end if
    end do

    print *,'cont',info

    !print *,'s',s(1,:)
    !print *,'y',y(1,:)
    if (info .EQ. 1) then
        if (istep .LE. k) then
            !no shuffling needed
            s(:,istep) = d_x
            y(:,istep) = d_jac
        else
            do i=2,k 
                s(:,i-1) = s(:,i)
                y(:,i-1) = y(:,i)
            end do
        s(:,k) = d_x
        y(:,k) = d_jac
        end if
    end if

    !print *,'s',s(1,:)
    !print *,'y',y(1,:)
!    print *,' '

END SUBROUTINE update_storage 
