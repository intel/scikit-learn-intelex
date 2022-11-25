/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#if 0
// custom distribution class for logistic regression
template<>
template< typename fptype, daal::algorithms::logistic_regression::training::Method method >
dist_custom< logistic_regression_training_manager< fptype, method > >
{
public:
    typedef logistic_regression_training_manager< fptype, method > Algo;

    struct LBFGSState
    {
        size_t m;
        size_t mc; // current size of the table
        size_t p;
        size_t index_start;
        std::vector<std::vector<fptype> > s;
        std::vector<std::vector<fptype> > y;
        std::vector<fptype> rho;

        LBFGSState(size_t table_size, size_t n_features, size_t _index_start=0)
            : m(table_size),
              mc(0),
              p(n_features),
              index_start(_index_start)
        {
            this->s.resize(this->m, std::vector<fptype>(this->p, 0));
            this->y.resize(this->m, std::vector<fptype>(this->p, 0));
            this->rho.resize(this->m, 0);
        }
    };

    // Compute dot(H0, q)
    //FIXME
    static void apply_initial_Hessian(T H0, S q)
    {
    #if 0
        if isinstance(H0, np.ndarray):
            if H0.ndim == 2:
                r = np.dot(H0, q)
            elif H0.ndim == 1:
                r = H0 * q
            else:
                raise ValueError("H0 is expected to be an ndarray of rank 1, 2 or a numpy scalar")
        elif np.isscalar(H0):
            r = H0 * q
        else:
            raise ValueError("Unexpected input H0")
    #endif // 0
                return r;
    }

    #TODO : implement ability to specify initial approximation to a Hessian, i.e.solve B.r = q
    #say for a sparse, or tridiagonal array

    // FIXME
    static void two_loop_recursion(std::vector<fptype> grad,
                                   std::vector<fptype> H0,
                                   size_t m,
                                   size_t index_start,
                                   std::vector<fptype> rho,
                                   std::vector<std::vector<fptype> > s,
                                   std::vector<std::vector<fptype> > y)
    {
        /*
        Computes H.grad, where H is stored per L_BFGS scheme,
        in vector rho, matrices s and y of outer dimensions m each.

        0 <= index start < table_size corresponds to position of the
        most recent vectore, and cyclic indexing is used.

        That is y[(index_start + i) % table_size] corresponds to `y_{k-1-i}` in
        the algorithm

        m - the number of actual data points stored in s, y and rho.
        */
        std::vector<fptype> q(grad);
        std::vector<fptype> alpha(rho);
        std::vector<fptype> buf(s[index_start]);
        size_t table_size = rho.size();
        for(size_t j=0; j<m; ++j) {
            auto i = (index_start + j) % table_size; // k - 1 - j
            np.dot(s[np.newaxis, i], q, out=alpha[i:i+1])
            alpha[i] *= rho[i]
            np.multiply(alpha[i], y[i], out=buf)
            q -= buf
                }

        r = apply_initial_Hessian(H0, q)

        for(size_t j=0; j<m; ++j) {
            i = (m - 1 - j + index_start) % table_size  # k - m + j == k - 1 - (m - 1 - j)
            beta = rho[i] * np.dot(y[i], r)
            np.multiply(alpha[i] - beta, s[i], out=buf)
            r += buf
        return r


    def compute_H_naive(grad, H0, m, index_start, rho, s, y):
            // Same as above, but straight-forward
        p = grad.shape[0]
        table_size = rho.shape[0]
        if isinstance(H0, np.ndarray) and H0.ndim == 2:
            H = H0
        elif np.isscalar(H0):
            H = H0 * np.eye(p, dtype=grad.dtype)
        for(size_t j=0; j<m; ++j) {
            i = (index_start + m - 1 - j) % table_size
            Vk = np.eye(p) - rho[i] * np.kron(y[i], s[i]).reshape((p,p))
            H = np.dot(Vk.T, H).dot(Vk)
            H += rho[i] * np.kron(s[i], s[i]).reshape((p,p))
        return np.dot(H, grad)


    def choose_H0(s, y, m, index_start):
        if m > 0:
            i = index_start
            yi = y[i]
            return np.dot(s[i], yi) / np.dot(yi, yi)
        else:
            return 1.0


    def line_search(x, fv, fg, dx, func, args):
                /*
                  Find step-length satisfying strong Wolfe's condition.

        Input:
            x  : function argument
            fv : function value f(x)
            fg : derivative f'(x)
            dx : proposed change of argument
          func : function to evaluate f and f'
          args : additional arguments to pass to func

        Output:
            stepLength : multiple of dx by which to move the argument
                */
        term1 = np.dot(fg, dx)
        stepLength = 1.0
        c1 = 0.0001  # Nocendal, Wright, ch. 3.1, between eqs. 3.4 and 3.5
        c2 = 0.9
        xn = x.copy()
        it = 0
        while stepLength > 0.0:
    #can use axpy ?
            np.multiply(stepLength, dx, out=xn)
            xn += x
            fv_new, fg_new = func(xn, *args)
            if fv_new - fv <= c1 * stepLength * term1:
                if abs(np.dot(fg_new, dx)) <= c2 * abs(term1) or np.all(fg_new == fg):
                    break
            it += 1
            stepLength *= 0.5 + 0.4 / it # 0.9, 0.7, 0.63, 0.6, ...
        assert stepLength > 0
        return stepLength # , fv_new, fg_new


    def deterministic_l_bfgs(func, x0, args=(), tol = 1e-8, l_bfgs_state=None, max_iters=100):
            /*
        Returns: (x_opt, f_val, f_grad, it, l_bfgs_state)
            */
        k = 0                 # index of iteration
        xc = np.asarray(x0)
        xn = np.empty_like(xc)
        p = x0.shape[0]

        if not isinstance(l_bfgs_state, LBFGSState):
            l_bfgs_state = LBFGSState(10, p,
                                      dtype = x0.dtype,
                                      index_start=0)

        m = l_bfgs_state.m
        mc = l_bfgs_state.mc
        index_start = l_bfgs_state.index_start
        s = l_bfgs_state.s
        y = l_bfgs_state.y
        rho = l_bfgs_state.rho

        in_convergence_basin = False
        f_val, f_grad = func(xc, *args)
        it = 0
        while it < max_iters:
            if np.abs(f_grad).max() < tol:
                break
            H0 = choose_H0(s, y, mc, index_start)
            r = two_loop_recursion(f_grad, H0, mc, index_start, rho, s, y)
            xn[:] = xc
            if in_convergence_basin:
                mdx = r
            else:
                // check Wolfe's condition
                stepLength = 1.0
                stepLength = line_search(xc, f_val, f_grad, -r, func, args)
                if stepLength == 1.0:
                    in_convergence_basin = True
                    mdx = r
                else:
                    mdx = stepLength * r
            xn -= mdx
            sn = -mdx
            f_val_next, f_grad_next = func(xn, *args)
            yn = f_grad_next - f_grad
            rhon = np.dot(sn, yn)
            if rhon > 0:
                rhon = 1.0 / rhon
                if (mc < m):
                    mc += 1
                if index_start > 0:
                    index_start -= 1
                else:
                    index_start = m - 1
                rho[index_start] = rhon
                y[index_start, :] = yn
                s[index_start, :] = sn
            f_grad = f_grad_next
            f_val = f_val_next
            xc = xn
            it += 1

        l_bfgs_state.m = m
        l_bfgs_state.mc = mc
        l_bfgs_state.index_start = index_start
        l_bfgs_state.s = s
        l_bfgs_state.y = y
        l_bfgs_state.rho = rho

        return (xc, f_val, f_grad, it, l_bfgs_state)

#endif // 0
