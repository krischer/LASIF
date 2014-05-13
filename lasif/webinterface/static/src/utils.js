if (typeof String.prototype.startsWith != 'function') {
  String.prototype.startsWith = function (str){
      return this.lastIndexOf(str, 0) === 0;
  };
}

if (typeof String.prototype.endsWith != 'function') {
    String.prototype.endsWith = function (str){
        return this.indexOf(str,this.length-str.length) !== -1;
    };
}