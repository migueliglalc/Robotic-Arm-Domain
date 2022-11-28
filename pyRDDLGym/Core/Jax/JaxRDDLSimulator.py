import jax
from typing import Dict

from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidExpressionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.Jax.JaxRDDLCompiler import JaxRDDLCompiler
from pyRDDLGym.Core.Parser.expr import Value
from pyRDDLGym.Core.Parser.rddl import RDDL
from pyRDDLGym.Core.Simulator.RDDLSimulator import RDDLSimulator

Args = Dict[str, Value]


class JaxRDDLSimulator(RDDLSimulator):
        
    def __init__(self,
                 rddl: RDDL,
                 key: jax.random.PRNGKey,
                 raise_error: bool=False, 
                 **compiler_args) -> None:
        self.rddl = rddl
        self.key = key
        self.raise_error = raise_error
        
        compiled = JaxRDDLCompiler(rddl, **compiler_args)
        compiled.compile()
        self.compiled = compiled
        
        self.invariants = jax.tree_map(jax.jit, compiled.invariants)
        self.preconds = jax.tree_map(jax.jit, compiled.preconditions)
        self.terminals = jax.tree_map(jax.jit, compiled.termination)
        self.reward = jax.jit(compiled.reward)
        self.cpfs = jax.tree_map(jax.jit, compiled.cpfs)
        
    @staticmethod
    def _print_stack_trace(expr, subs, key):
        return str(jax.make_jaxpr(expr)(subs, key))
    
    def handle_error_code(self, error, msg):
        if self.raise_error:
            errors = JaxRDDLCompiler.get_error_messages(error)
            if errors:
                errors = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(errors))
                message = f'Internal error in evaluation of {msg}:\n'
                raise RDDLInvalidExpressionError(message + errors)
    
    def _check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for i, invariant in enumerate(self.invariants):
            sample, self.key, error = invariant(self.subs, self.key)
            self.handle_error_code(error, f'invariant {i + 1}')
            
            if not bool(sample):
                raise RDDLStateInvariantNotSatisfiedError(
                    f'Invariant {i + 1} is not satisfied.\n' + 
                    JaxRDDLSimulator._print_stack_trace(
                        invariant, self.subs, self.key))
    
    def check_action_preconditions(self) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        for i, precond in enumerate(self.preconds):
            sample, self.key, error = precond(self.subs, self.key)
            self.handle_error_code(error, f'precondition {i + 1}')
            
            if not bool(sample):
                raise RDDLActionPreconditionNotSatisfiedError(
                    f'Precondition {i + 1} is not satisfied.\n' + 
                    JaxRDDLSimulator._print_stack_trace(
                        precond, self.subs, self.key))
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for i, terminal in enumerate(self.terminals):
            sample, self.key, error = terminal(self.subs, self.key)
            self.handle_error_code(error, f'termination {i + 1}')
            
            if bool(sample):
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        reward, self.key, error = self.reward(self.subs, self.key)
        self.handle_error_code(error, 'reward function')
        return float(reward)
    
    def reset(self) -> Args:
        '''Resets the state variables to their initial values.'''
        self.subs = self.compiled.init_values.copy()  
        obs = {state: self.subs[state] 
               for state in self.compiled.next_states.values()}        
        done = self.check_terminal_states()        
        return obs, done
    
    def step(self, actions: Args) -> Args:
        '''Samples and returns the next state from the cpfs.
        
        :param actions: a dict mapping current action fluents to their values
        '''
        subs = self.subs
        subs.update(actions)
        
        for name, cpf in self.cpfs.items():
            subs[name], self.key, error = cpf(subs, self.key)
            self.handle_error_code(error, f'CPF <{name}>')
            
        reward = self.sample_reward()
        
        obs = {}
        for next_state, state in self.compiled.next_states.items():
            obs[state] = subs[state] = subs[next_state]
        
        done = self.check_terminal_states()
        
        return obs, reward, done
        
