'''
 @Author: Emanuele Sansone 
 @Date: 2017-04-18 08:52:29 
 @Last Modified by: Emanuele Sansone 
 @Last Modified time: 2017-04-18 08:52:29
'''

import __future__ 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.animation as animation
import seaborn
from scipy.stats import norm
import os

GAMES = 500
DISCR_UPDATE = 1
GEN_UPDATE = 50

# Training data
class RealDistribution:
    def __init__(self):
        self.mu = 5
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        return samples

# Noise data
class NoiseDistribution:
    def __init__(self):
        self.low = 0
        self.high = 1

    def sample(self, N):
        samples = np.random.uniform(self.low, self.high, N)
        return samples

# GAN
class GAN:
    def __init__(self):
        self.games = GAMES
        self.discriminator_steps = DISCR_UPDATE
        self.generator_steps = GEN_UPDATE
        self.learning_rate = 0.1
        self.num_samples = 10
        self.skip_log = 20

        self.noise = NoiseDistribution()
        self.data = RealDistribution()

        self.create_model()

    def linear(self, input, scope=None):
        init_w = tf.random_normal_initializer(stddev=0.1)
        init_b = tf.constant_initializer(0.0)
        with tf.variable_scope(scope or 'linear'):              # USING SCOPE FOR FUTURE VERSION WITH MULTIPLE LAYERS
            w = tf.get_variable('w', [1,1], initializer=init_w)
            b = tf.get_variable('b', [1,1], initializer=init_b)
            return tf.add(tf.matmul(w, input), b)

    def generator(self, input):
        logits = self.linear(input, 'gen')
        return logits

    def discriminator(self, input):
        logits = self.linear(input, 'discr')
        pred = tf.sigmoid(logits)
        return pred

    def create_model(self):
        # Generator
        self.epsilon = tf.placeholder(tf.float32, shape=(1, self.num_samples))

        with tf.variable_scope('GEN'):
            self.z = tf.placeholder(tf.float32, shape=(1, self.num_samples))
            self.gen = tf.add(self.generator(self.z), self.epsilon)

        # Discriminator
        with tf.variable_scope('DISC') as scope:
            self.x = tf.placeholder(tf.float32, shape=(1, self.num_samples))
            self.x_noisy = tf.add(self.x, self.epsilon)
            self.discr1 = self.discriminator(self.x_noisy)
            scope.reuse_variables()
            self.discr2 = self.discriminator(self.gen)

        # Losses
        self.loss_gen = tf.reduce_mean(tf.log(1-self.discr2))
        self.loss_discr = tf.reduce_mean(-tf.log(self.discr1) -tf.log(1-self.discr2))

        # Parameters
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.discr_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DISC')
        self.all_params = tf.trainable_variables()

        # Optimizers
        self.opt_gen = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss_gen,
            var_list=self.gen_params
        )
        self.opt_discr = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss_discr,
            var_list=self.discr_params
        )

        # Gradients
        self.grad_discr = tf.gradients(self.loss_discr, self.discr_params)[0]
        self.grad_gen = tf.gradients(self.loss_gen, self.gen_params)[0]

        # Hessian computation
        hessian = []
        for v1 in self.all_params:
            temp = []
            for v2 in self.all_params:
                # computing derivative twice, first w.r.t v2 and then w.r.t v1
                temp.append(tf.gradients(tf.gradients(-self.loss_discr, v2)[0], v1)[0])
            temp = [tf.constant(0, dtype=tf.float32) if t == None else t for t in temp] # tensorflow returns None when there is no gradient, so we replace None with 0
            temp = tf.stack(temp)
            hessian.append(temp)
        self.hessian = tf.squeeze(tf.stack(hessian))

        
    def train(self):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            x = self.data.sample(self.num_samples)
            objective_function = []
            grad_magn_discr = []
            grad_magn_gen = []
            eigs = []
            frames = 0

            for games in range(self.games):
                eps_noise = np.random.normal(0, 1, self.num_samples)

                # Update discriminator
                z = self.noise.sample(self.num_samples)
                for discr_steps in range(self.discriminator_steps):
                    loss_discr, _ = sess.run([self.loss_discr, self.opt_discr],{
                        self.x: np.reshape(x, (1,self.num_samples)),
                        self.z: np.reshape(z, (1,self.num_samples)),
                        self.epsilon: np.reshape(eps_noise, (1,self.num_samples))
                    })
                grad_discr_val = sess.run(self.grad_discr, feed_dict={
                        self.x: np.reshape(x, (1,self.num_samples)),
                        self.z: np.reshape(z, (1,self.num_samples)),
                        self.epsilon: np.reshape(eps_noise, (1,self.num_samples))
                    })
                grad_magn_discr.append(np.linalg.norm(grad_discr_val))

                # Intermediate visualization
                if games % self.skip_log == 0:
                    print('game %d: Loss: %.3f\tTarget loss: %.3f' % (games, -loss_discr, -2*np.log(2)))
                    self.intuition(sess, x)
                    frame = plt.gca()
                    frame.axes.get_yaxis().set_visible(False)
                    plt.draw()
                    plt.pause(0.01)
                    plt.clf()

                # Update generator
                for gen_steps in range(self.generator_steps):
                    z = self.noise.sample(self.num_samples)
                    loss_gen, _ = sess.run([self.loss_gen, self.opt_gen],{
                        self.z: np.reshape(z, (1,self.num_samples)),
                        self.epsilon: np.reshape(eps_noise, (1,self.num_samples))
                    })
                grad_gen_val = sess.run(self.grad_gen, feed_dict={
                        self.z: np.reshape(z, (1,self.num_samples)),
                        self.epsilon: np.reshape(eps_noise, (1,self.num_samples))
                    })
                grad_magn_gen.append(np.linalg.norm(grad_gen_val))

                # Hessian computation
                hessian_eig = []
                hess_val = sess.run(self.hessian, feed_dict={
                        self.x: np.reshape(x, (1,self.num_samples)),
                        self.z: np.reshape(z, (1,self.num_samples)),
                        self.epsilon: np.reshape(eps_noise, (1,self.num_samples))
                    })
                hessian_eig, _ = np.linalg.eig(hess_val)
                eigs.append(hessian_eig)

                # Intermediate visualization
                if games % self.skip_log == 0:
                    print('game %d: Loss: %.3f\tTarget loss: %.3f' % (games, -loss_discr, -2*np.log(2)))
                    self.intuition(sess, x)
                    frame = plt.gca()
                    frame.axes.get_yaxis().set_visible(False)
                    plt.savefig('img/img_noisy-'+str(games)+'.png')
                    plt.draw()
                    plt.pause(0.01)
                    plt.clf()
                    frames += 1
                
                objective_function.append(-loss_discr)

            # Visualization
            plt.close()
            print('\nSaving summary...\n')
            gs = gridspec.GridSpec(2, 2)

            # Graphical interpretation
            plt.subplot(gs[0,0])
            self.intuition(sess, x)
            frame = plt.gca()
            frame.axes.get_yaxis().set_visible(False)

            # Objective function
            plt.subplot(gs[0,1])
            self.objective(objective_function, games)

            # Gradient discriminator
            plt.subplot(gs[1,0])
            plt.plot(range(self.games),grad_magn_discr)
            plt.title('Gradient magnitude - Discriminator')

            # Gradient generator
            plt.subplot(gs[1,1])
            plt.plot(range(self.games),grad_magn_gen)
            plt.title('Gradient magnitude - Generator')
            plt.savefig('img/summary_noisy_'+str(self.games)+'_'+str(self.discriminator_steps)+\
                      '_'+str(self.generator_steps)+'.eps')
            plt.savefig('img/summary_noisy_'+str(self.games)+'_'+str(self.discriminator_steps)+\
                      '_'+str(self.generator_steps)+'.png')
 
            # Eigenvalues of Hessian
            print('\nSaving the eigenvalues of the Hessian...\n')
            eigs = np.array(eigs)
            plt.figure(2)
            plt.plot(range(self.games),eigs[:,0])
            plt.plot(range(self.games),eigs[:,1])
            plt.plot(range(self.games),eigs[:,2])
            plt.plot(range(self.games),eigs[:,3])
            plt.title('Eigenvalues of Hessian vs. Iterations (Symlog scale)')
            plt.yscale('symlog')
            plt.savefig('img/hessian_noisy_'+str(self.games)+'_'+str(self.discriminator_steps)+\
                      '_'+str(self.generator_steps)+'.eps')
            plt.savefig('img/hessian_noisy_'+str(self.games)+'_'+str(self.discriminator_steps)+\
                      '_'+str(self.generator_steps)+'.png')
            
            # Animation
            print('\nCreating GIF animation...')
            fig = plt.figure()
            plt.axis('off')
            anim = animation.FuncAnimation(fig, self.animate, frames=frames)
            anim.save('img/img_noisy_'+str(self.games)+'_'+str(self.discriminator_steps)+\
                      '_'+str(self.generator_steps)+'.gif', writer='imagemagick', fps=int(120/self.skip_log))
            self.delete()


    def animate(self, i):
        print('Frame {}'.format(i))
        img = mpimg.imread('img/img_noisy-'+str(i*self.skip_log)+'.png')
        ax = plt.imshow(img)
        return ax

    def delete(self):
        i = 0
        while True:
            try:
                os.remove('img/img_noisy-'+str(i*self.skip_log)+'.png')
                i += 1
            except:
                return

    def intuition(self, sess, x):
        min_range = self.noise.low
        max_range = self.data.mu+2*self.data.sigma
        plt.xlim([min_range,max_range])
        plt.ylim([-0.6,1])

        # Lines
        plt.plot([min_range, max_range], [-0.5,-0.5], 'k-', lw=1)
        plt.plot([min_range, max_range], [0,0], 'k-', lw=1)

        # Samples
        num = 10
        z = self.noise.sample(num)
        plt.plot(z, -0.5*np.ones(num),'bo')
        out = sess.run(self.gen, {self.z: np.reshape(z, (1,self.num_samples)),
                                  self.epsilon: np.reshape(np.zeros(self.num_samples), (1,self.num_samples))
                                 })
        plt.plot(np.transpose(out), \
                np.zeros(num),'bo')
            
        # Arrows
        for i in range(num):
            plt.plot([z[i],out[0][i]],[-0.49,-0.01],'-k')

        # Real distribution
        x_range = np.linspace(min_range, max_range, 50)
        fit = norm.pdf(x_range, self.data.mu, self.data.sigma)
        plt.plot(x_range, fit, '-g')           

        # Real data
        plt.plot(x, np.zeros(self.num_samples),'go')

        # Discriminator
        num = 40*self.num_samples
        x_range = np.linspace(min_range, max_range, num)
        out = []
        for i in range(int(num/self.num_samples)):
            tmp = x_range[i*self.num_samples:(i+1)*self.num_samples]
            tmp = sess.run(self.discr1, {self.x: np.reshape(tmp, (1,self.num_samples)),
                                         self.epsilon: np.reshape(np.zeros(self.num_samples), (1,self.num_samples))
                                        })[0]
            for j in range(self.num_samples):
                out.append(tmp[j])
        plt.plot(x_range, \
                np.array(out),'-b')
        
        plt.title('Graphical interpretation')

    def objective(self, objective, games):
        plt.plot(range(self.games),objective)
        plt.plot([1, games], [-2*np.log(2), -2*np.log(2)], 'r-', lw=1)
        plt.title('Objective vs. Iterations')


if __name__ == '__main__':
    model = GAN()
    model.train()
